"""
Templar TPS Evaluation Environment (Validator-Owned)

URL-Based Architecture:
- This env.py is owned by the VALIDATOR, not the miner
- Miner commits a URL pointing to their train.py code
- Validator downloads code from URL and passes it to this env
- This ensures miners can't tamper with evaluation logic

Flow:
1. Validator reads miner commitment (contains code URL)
2. Validator downloads train.py from URL
3. Validator calls env.evaluate(code="...", ...)
4. This Actor runs benchmark and returns MFU
"""

import ast
import asyncio
import gc
import hashlib
import importlib.util
import logging
import math
import os
import random
import sys
import time
import traceback
import uuid as _uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from crusades.core.security_defs import (
    _MAX_BYTES_LITERAL_ELTS,
    ALLOWED_TORCH_ASSIGNMENT_PREFIXES,
    ALLOWED_TORCH_SUBMODULE_IMPORTS,
    FORBIDDEN_ASSIGNMENT_ATTRS,
    FORBIDDEN_ATTR_CALLS,
    FORBIDDEN_BACKEND_TOGGLE_ATTRS,
    FORBIDDEN_BUILTINS,
    FORBIDDEN_CUDNN_ATTRS,
    FORBIDDEN_DIRECT_CALLS,
    FORBIDDEN_DOTTED_MODULES,
    FORBIDDEN_GC_ATTRS,
    FORBIDDEN_GRAD_TOGGLE_CALLS,
    FORBIDDEN_IMPORT_SUBSTRINGS,
    FORBIDDEN_INTROSPECTION_ATTRS,
    FORBIDDEN_MODULES,
    FORBIDDEN_NAMES,
    FORBIDDEN_OBJECT_DUNDER_ATTRS,
    FORBIDDEN_STRINGS,
    FORBIDDEN_SYS_MODULE_NAMES,
    FORBIDDEN_TIMER_ATTRS,
    FORBIDDEN_TORCH_ATTRIBUTE_ALIASES,
    FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS,
    FORBIDDEN_TORCH_CONFIG_MODULES,
    FORBIDDEN_TORCH_SYMBOL_IMPORTS,
    SuspiciousConstructionError,
    forbidden_name_binding_reason,
    try_decode_bytes_node,
    try_decode_str_bytes_constructor,
    try_resolve_concat,
    try_resolve_format,
    try_resolve_fstring,
    try_resolve_join,
)

# Setup logging - only for this module, don't configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)

# Suppress noisy loggers from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

_perf_counter = time.perf_counter
_monotonic = time.monotonic
_cuda_synchronize = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
_cuda_elapsed_time = torch.cuda.Event.elapsed_time if torch.cuda.is_available() else None

# Multi-GPU rank detection (set by torchrun)
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_IS_RANK_0 = _LOCAL_RANK == 0

# Save the REAL torch.nn.functional functions at import time, before any miner
# code can monkey-patch them.  The reference implementation uses these saved
# copies exclusively so that ``F.cross_entropy = fake`` cannot poison the
# reference run.
_real_cross_entropy = F.cross_entropy
_REAL_CE_ID = id(_real_cross_entropy)

# Immutable identity fingerprints of the real timer functions, captured at import
# time before any miner code runs.  Used by the runtime tamper check.
_REAL_PC_ID = id(time.perf_counter)
_REAL_MONO_ID = id(time.monotonic)
_REAL_SYNC_ID = id(_cuda_synchronize)
_REAL_ET_ID = id(_cuda_elapsed_time) if _cuda_elapsed_time is not None else None


def _make_timer_vault():
    """Immutable timer references stored in closure variables.

    Stack-walking attacks (inspect.currentframe / frame.f_back) can find and
    patch module-level globals like _perf_counter and _REAL_PC_ID.  Closure
    variables are NOT accessible via frame.f_globals, making them immune to
    this class of attack.
    """
    pc = time.perf_counter
    mo = time.monotonic
    sy = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
    et = torch.cuda.Event.elapsed_time if torch.cuda.is_available() else None
    ce = F.cross_entropy
    pc_id, mo_id, sy_id = id(pc), id(mo), id(sy)
    et_id = id(et) if et is not None else None
    ce_id = id(ce)

    def get_timers():
        return pc, mo, sy, et, ce

    def get_real_ids():
        return pc_id, mo_id, sy_id, et_id, ce_id

    return get_timers, get_real_ids


_vault_get_timers, _vault_get_real_ids = _make_timer_vault()

# Configuration from environment variables
DETERMINISTIC_MODE = os.getenv("DETERMINISTIC_MODE", "1") == "1"
try:
    EVAL_SEQUENCE_LENGTH = int(os.getenv("EVAL_SEQUENCE_LENGTH", "1024"))
except ValueError:
    logger.warning("Invalid EVAL_SEQUENCE_LENGTH env var, defaulting to 1024")
    EVAL_SEQUENCE_LENGTH = 1024
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/templar_eval"))


@dataclass
class InnerStepsResult:
    """Expected return type from miner's inner_steps function."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None


def _log_vram(tag: str):
    """Log current VRAM usage for debugging memory leaks."""
    if not torch.cuda.is_available():
        return
    dev = _LOCAL_RANK
    allocated = torch.cuda.memory_allocated(dev) / 1024**3
    reserved = torch.cuda.memory_reserved(dev) / 1024**3
    logger.info(f"[VRAM {tag}] rank={dev} allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


# Global cache for model (data is NOT cached for validators)
_CACHE = {
    "model": None,
    "model_path": None,
    "initial_state": None,
    "use_random_init": None,
}


@contextmanager
def _hide_sensitive_env_modules():
    """Temporarily hide env modules during miner code execution."""
    hidden_modules = {}
    for module_key in list(sys.modules):
        if module_key == "env" or module_key == __name__ or module_key.endswith(".env"):
            hidden_modules[module_key] = sys.modules.pop(module_key)
    try:
        yield
    finally:
        sys.modules.update(hidden_modules)


_VALID_STRATEGIES = ("ddp", "fsdp", "tp")


@dataclass
class ParallelismConfig:
    """Parallelism topology declared by a miner's ``get_strategy()``."""

    dp_size: int
    tp_size: int


def _detect_strategy_from_source(source: str, num_gpus: int = 1) -> ParallelismConfig | None:
    """Extract parallelism topology from miner source via AST (no code execution).

    Supports two ``get_strategy()`` return formats:

    - **Legacy string**: ``return "ddp"`` / ``"fsdp"`` / ``"tp"`` — converted
      to ``ParallelismConfig`` using *num_gpus*.
    - **Dict literal**: ``return {"dp_size": 2, "tp_size": 2}`` — parsed
      directly.  Only simple ``ast.Constant`` keys/values are accepted.

    Returns ``None`` when the function is absent, unparseable, or contains
    non-literal expressions.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "get_strategy"):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Return) or stmt.value is None:
                continue

            # Legacy string format: "ddp" / "fsdp" / "tp"
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                val = stmt.value.value.lower()
                if val in _VALID_STRATEGIES:
                    if val in ("ddp", "fsdp"):
                        return ParallelismConfig(dp_size=num_gpus, tp_size=1)
                    return ParallelismConfig(dp_size=1, tp_size=num_gpus)

            # Dict literal format: {"dp_size": int, "tp_size": int}
            if isinstance(stmt.value, ast.Dict):
                d: dict[str, int] = {}
                for k, v in zip(stmt.value.keys, stmt.value.values):
                    if (
                        isinstance(k, ast.Constant)
                        and isinstance(k.value, str)
                        and isinstance(v, ast.Constant)
                        and isinstance(v.value, int)
                    ):
                        d[k.value] = v.value
                if "dp_size" in d and "tp_size" in d:
                    dp = d["dp_size"]
                    tp = d["tp_size"]
                    if dp >= 1 and tp >= 1:
                        return ParallelismConfig(dp_size=dp, tp_size=tp)

    return None


def _load_miner_module(train_path: Path):
    """Load miner's train.py as a module in an isolated context."""
    spec = importlib.util.spec_from_file_location("miner_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {train_path}")

    with _hide_sensitive_env_modules():
        module = importlib.util.module_from_spec(spec)
        sys.modules["miner_train"] = module
        spec.loader.exec_module(module)

    return module


def _reset_torch_state():
    """Reset torch global state between evaluations.

    Restores timer/function references from the closure vault rather than from
    module globals.  This makes the reset self-healing: even if a miner managed
    to mutate the module-level ``_perf_counter`` etc. via ``frame.f_globals``,
    the vault values are immune (closure variables are not reachable through
    frame introspection) and will restore the worker to a clean state.
    """
    global _perf_counter, _monotonic, _cuda_synchronize, _cuda_elapsed_time, _real_cross_entropy
    global _REAL_PC_ID, _REAL_MONO_ID, _REAL_SYNC_ID, _REAL_ET_ID, _REAL_CE_ID

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    _enforce_backend_state()
    torch.set_grad_enabled(True)
    torch.set_default_dtype(torch.float32)

    if "miner_train" in sys.modules:
        del sys.modules["miner_train"]

    vpc, vmo, vsy, vet, vce = _vault_get_timers()

    # Restore the actual library-level entry points from vault
    time.perf_counter = vpc
    time.monotonic = vmo
    F.cross_entropy = vce
    if torch.cuda.is_available():
        torch.cuda.synchronize = vsy
        if vet is not None:
            torch.cuda.Event.elapsed_time = vet

    # Refresh module globals so they stay in sync with the vault.
    # Without this, a poisoned global would persist and diverge from
    # the (clean) library-level function, causing the next runtime
    # integrity check to flag a mismatch.
    _perf_counter = vpc
    _monotonic = vmo
    _real_cross_entropy = vce
    _cuda_synchronize = vsy
    _cuda_elapsed_time = vet

    _REAL_PC_ID = id(vpc)
    _REAL_MONO_ID = id(vmo)
    _REAL_SYNC_ID = id(vsy)
    _REAL_ET_ID = id(vet) if vet is not None else None
    _REAL_CE_ID = id(vce)

    logger.debug("Torch state reset complete (vault-sourced)")


def _enforce_backend_state():
    """Set torch backend settings to canonical values."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def _check_backend_state() -> list[str]:
    """Check torch backend settings. Returns list of violations."""
    violations = []
    if torch.backends.cudnn.deterministic:
        violations.append("cudnn.deterministic")
    if not torch.backends.cudnn.benchmark:
        violations.append("cudnn.benchmark")
    if not torch.backends.cudnn.allow_tf32:
        violations.append("cudnn.allow_tf32")
    if torch.get_float32_matmul_precision() != "high":
        violations.append("float32_matmul_precision")
    if not torch.backends.cuda.flash_sdp_enabled():
        violations.append("flash_sdp_enabled")
    if not torch.backends.cuda.mem_efficient_sdp_enabled():
        violations.append("mem_efficient_sdp_enabled")
    if not torch.backends.cuda.math_sdp_enabled():
        violations.append("math_sdp_enabled")
    return violations


def _load_hf_dataset(
    dataset_name: str,
    model_name: str,
    num_samples: int = 10000,
    sequence_length: int = 1024,
    split: str = "train",
    validator_seed: str | None = None,
) -> torch.Tensor:
    """Load and tokenize dataset from HuggingFace or local cache.

    When running with --network none, uses pre-cached dataset from Docker image.
    The validator seed is used to shuffle the cached data.

    Args:
        dataset_name: HuggingFace dataset name
        model_name: Model name for tokenizer
        num_samples: Number of samples to load
        sequence_length: Sequence length
        split: Dataset split
        validator_seed: Seed string from validator (for unpredictable sampling)

    Returns:
        Tensor of shape [num_samples, sequence_length]
    """
    import json

    from transformers import AutoTokenizer

    # Determine seed: validators use unpredictable seed
    if validator_seed:
        seed_hash = hashlib.sha256(validator_seed.encode()).hexdigest()
        actual_seed = int(seed_hash[:8], 16)
        logger.info(f"Validator mode: seed={actual_seed} (from {validator_seed})")
    else:
        actual_seed = 42
        logger.info(f"Test mode: fixed seed={actual_seed}")

    logger.info(f"Loading dataset: {dataset_name} (samples={num_samples})")

    # Cache tokenizer to avoid reloading every eval (HF tokenizers leak via Rust FFI)
    if _CACHE.get("tokenizer") is not None and _CACHE.get("tokenizer_model") == model_name:
        tokenizer = _CACHE["tokenizer"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _CACHE["tokenizer"] = tokenizer
        _CACHE["tokenizer_model"] = model_name

    # Check for cached dataset (enables --network none operation)
    cached_path = os.getenv("CACHED_DATASET_PATH", "/home/appuser/.cache/templar/dataset.json")

    if Path(cached_path).exists():
        # Tokenize ALL samples once and cache the full tensor.
        # Subsequent evals just shuffle indices (fast) instead of
        # re-tokenizing 10k samples (slow + leaks memory).
        cache_key = f"tokenized_{cached_path}_{sequence_length}"
        all_tokenized = _CACHE.get(cache_key)

        if all_tokenized is None:
            logger.info(f"First load — tokenizing cached dataset: {cached_path}")
            with open(cached_path) as f:
                all_samples = json.load(f)

            tokens_list = []
            for text in all_samples:
                encoded = tokenizer(
                    text,
                    max_length=sequence_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                tokens_list.append(encoded["input_ids"].squeeze(0))

            if not tokens_list:
                raise ValueError("No samples in cached dataset")

            all_tokenized = torch.stack(tokens_list)
            _CACHE[cache_key] = all_tokenized
            logger.info(f"Tokenized and cached: {all_tokenized.shape}")
        else:
            logger.info(f"Using pre-tokenized cache: {all_tokenized.shape}")

        # Shuffle indices with validator seed for unpredictability
        total = all_tokenized.size(0)
        indices = list(range(total))
        rng = random.Random(actual_seed)
        rng.shuffle(indices)
        selected = indices[:num_samples]

        data = all_tokenized[selected]
        logger.info(f"Sampled data: shape={data.shape}, seed={actual_seed}")
        return data

    # Fallback: Load from HuggingFace (requires network)
    logger.info("No cache found, loading from HuggingFace...")
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=actual_seed, buffer_size=10000)

    tokens_list = []
    dataset_iter = iter(dataset)

    for _ in range(num_samples):
        try:
            sample = next(dataset_iter)
            text = sample.get("text", sample.get("content", ""))

            encoded = tokenizer(
                text,
                max_length=sequence_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokens_list.append(encoded["input_ids"].squeeze(0))

        except StopIteration:
            break

    if not tokens_list:
        raise ValueError(f"No samples loaded from {dataset_name}")

    data = torch.stack(tokens_list)
    logger.info(f"Loaded data: shape={data.shape}, seed={actual_seed}")
    return data


def _set_deterministic(seed: int) -> None:
    """Set seed for reproducibility.

    Note: We rely on seed-based reproducibility (manual_seed), not cuDNN
    deterministic mode. cuDNN deterministic mode forces slower algorithms
    and is unnecessary for BF16 transformer training where compute is
    dominated by cuBLAS matmuls (not cuDNN convolutions).
    """
    if not DETERMINISTIC_MODE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_main_guard_nodes(tree: ast.AST) -> set[int]:
    """Collect AST node IDs inside `if __name__ == "__main__":` blocks."""
    skip_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            # Match: if __name__ == "__main__" or if "__main__" == __name__
            is_main_guard = False
            if isinstance(test, ast.Compare) and len(test.ops) == 1:
                op = test.ops[0]
                if isinstance(op, (ast.Eq, ast.Is)):
                    left, right = test.left, test.comparators[0]
                    if (
                        isinstance(left, ast.Name)
                        and left.id == "__name__"
                        and isinstance(right, ast.Constant)
                        and right.value == "__main__"
                    ) or (
                        isinstance(right, ast.Name)
                        and right.id == "__name__"
                        and isinstance(left, ast.Constant)
                        and left.value == "__main__"
                    ):
                        is_main_guard = True
            if is_main_guard:
                for child in ast.walk(node):
                    skip_ids.add(id(child))
    return skip_ids


def _scan_for_dangerous_patterns(tree: ast.AST) -> tuple[bool, str | None]:
    """AST scan to reject forbidden code patterns."""
    # All forbidden-pattern sets are defined in crusades.core.security_defs
    # and imported at module level.  Local aliases keep the code below concise.
    _forbidden_names = FORBIDDEN_NAMES
    _forbidden_modules = FORBIDDEN_MODULES
    _forbidden_dotted_modules = FORBIDDEN_DOTTED_MODULES
    _forbidden_torch_symbol_imports = FORBIDDEN_TORCH_SYMBOL_IMPORTS
    _forbidden_torch_backend_symbol_imports = FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS
    _forbidden_torch_attribute_aliases = FORBIDDEN_TORCH_ATTRIBUTE_ALIASES
    _forbidden_builtins = FORBIDDEN_BUILTINS

    # Track names that currently alias the torch module (e.g. "import torch as tl",
    # or "tl = torch"). This lets us block torch.load() even through aliases.
    torch_aliases = {"torch"}

    # Track names that alias torch submodules (e.g. "F" from
    # "import torch.nn.functional as F").  Attribute *assignment* on any
    # of these is forbidden — legitimate code never monkey-patches torch
    # module attributes (e.g. ``F.cross_entropy = fake``).
    torch_submodule_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    local_name = alias.asname or alias.name
                    if local_name != "torch":
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: aliasing torch is forbidden"
                    torch_aliases.add(local_name)
                elif alias.name.startswith("torch.") and alias.asname:
                    if alias.name in ALLOWED_TORCH_SUBMODULE_IMPORTS:
                        torch_submodule_aliases.add(alias.asname)

        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Name) and node.value.id in torch_aliases:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id != "torch":
                            line = getattr(node, "lineno", "?")
                            return False, f"Line {line}: aliasing torch is forbidden"
                        torch_aliases.add(target.id)

        # Walrus operator: if (t := torch): ... creates an untracked alias
        if isinstance(node, ast.NamedExpr):
            if isinstance(node.value, ast.Name) and node.value.id in torch_aliases:
                if isinstance(node.target, ast.Name) and node.target.id != "torch":
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: aliasing torch via walrus operator is forbidden"
            # ld := torch.load, dyn := torch._dynamo — attribute on torch alias
            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in torch_aliases
                and node.value.attr in _forbidden_torch_attribute_aliases
            ):
                line = getattr(node, "lineno", "?")
                return False, (
                    f"Line {line}: binding torch.{node.value.attr} via walrus operator is forbidden"
                )

        # Tuple/list unpacking: [t] = [torch], t, = torch, ...
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Tuple, ast.List)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Name) and elt.id in torch_aliases:
                    for target in node.targets:
                        names = []
                        if isinstance(target, (ast.Tuple, ast.List)):
                            names = [n.id for n in target.elts if isinstance(n, ast.Name)]
                        elif isinstance(target, ast.Name):
                            names = [target.id]
                        for n in names:
                            if n != "torch":
                                line = getattr(node, "lineno", "?")
                                return False, (
                                    f"Line {line}: aliasing torch via unpacking is forbidden"
                                )
                # ld, = [torch.load] — attribute on torch alias via unpacking
                if (
                    isinstance(elt, ast.Attribute)
                    and isinstance(elt.value, ast.Name)
                    and elt.value.id in torch_aliases
                    and elt.attr in _forbidden_torch_attribute_aliases
                ):
                    line = getattr(node, "lineno", "?")
                    return False, (
                        f"Line {line}: binding torch.{elt.attr} via unpacking is forbidden"
                    )

        # Block attribute mutation on torch modules / submodule aliases.
        # Covers Assign, AugAssign (+=), and Delete (del) targets.
        # Catches: F.cross_entropy = fake, torch.nn.functional.cross_entropy = fake,
        #          torch.autograd.backward = fake, torch.Tensor.backward = fake,
        #          del F.cross_entropy, F.cross_entropy += ..., etc.
        # Allows: torch.backends.cuda.matmul.allow_bf16_... = True (legitimate config).
        _attr_targets: list[ast.Attribute] = []
        if isinstance(node, ast.Assign):
            _attr_targets = [t for t in node.targets if isinstance(t, ast.Attribute)]
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute):
            _attr_targets = [node.target]
        elif isinstance(node, ast.Delete):
            _attr_targets = [t for t in node.targets if isinstance(t, ast.Attribute)]

        for target in _attr_targets:
            root_node = target.value
            while isinstance(root_node, ast.Attribute):
                root_node = root_node.value
            if not isinstance(root_node, ast.Name):
                continue
            root_name = root_node.id

            if root_name in torch_submodule_aliases:
                line = getattr(node, "lineno", "?")
                return (
                    False,
                    f"Line {line}: mutating {root_name}.{target.attr} is forbidden"
                    " (monkey-patching torch modules is not allowed)",
                )

            if root_name in torch_aliases:
                if isinstance(target.value, ast.Name):
                    line = getattr(node, "lineno", "?")
                    return (
                        False,
                        f"Line {line}: mutating {root_name}.{target.attr} is forbidden"
                        " (monkey-patching torch modules is not allowed)",
                    )
                # Nested path — block unless parent is an allowed config prefix
                parts: list[str] = []
                walk = target.value
                while isinstance(walk, ast.Attribute):
                    parts.append(walk.attr)
                    walk = walk.value
                if isinstance(walk, ast.Name):
                    parts.append(walk.id)
                    parent_path = ".".join(reversed(parts))
                    # Allow torch.backends.* config assignments
                    if any(
                        parent_path == pfx or parent_path.startswith(pfx + ".")
                        for pfx in ALLOWED_TORCH_ASSIGNMENT_PREFIXES
                    ):
                        continue
                    line = getattr(node, "lineno", "?")
                    return (
                        False,
                        f"Line {line}: mutating {parent_path}.{target.attr}"
                        " is forbidden (monkey-patching torch modules is not allowed)",
                    )

        # Block rebinding sensitive torch attributes to local names
        # (e.g. `c = torch.compile`, `ld = torch.load`, `dyn = torch._dynamo`)
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in torch_aliases
            and node.value.attr in _forbidden_torch_attribute_aliases
        ):
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: aliasing torch.{node.value.attr} is forbidden"

        # Block bare-name references to dangerous builtins (e.g. `_imp = __import__`)
        if isinstance(node, ast.Name) and node.id in _forbidden_names:
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: reference to '{node.id}' is forbidden"

        name_binding_violation = forbidden_name_binding_reason(node)
        if name_binding_violation:
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: {name_binding_violation}"

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_OBJECT_DUNDER_ATTRS:
            if isinstance(node.value, ast.Name) and node.value.id == "object":
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        # Block torch._C access (low-level C++ bindings escape hatch)
        if isinstance(node, ast.Attribute) and node.attr == "_C":
            if isinstance(node.value, ast.Name) and node.value.id in torch_aliases:
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: torch._C access is forbidden"

        # Block torch._dynamo.config and torch._inductor.config writes
        # (e.g. suppress_errors=True masks compilation errors)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute):
                    if target.value.attr == "config" and isinstance(
                        target.value.value, ast.Attribute
                    ):
                        if target.value.value.attr in FORBIDDEN_TORCH_CONFIG_MODULES:
                            line = getattr(node, "lineno", "?")
                            return (
                                False,
                                f"Line {line}: modifying torch.{target.value.value.attr}.config is forbidden",
                            )

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "__class__":
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden pattern detected"

        # Reading __class__ is allowed (needed for FSDP layer detection);
        # only *writing* to __class__ is blocked above.

        # Block timer-related attribute access on ANY object.
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_TIMER_ATTRS:
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in FORBIDDEN_ASSIGNMENT_ATTRS:
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr == "__slots__":
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_GC_ATTRS:
            if isinstance(node.value, ast.Name) and node.value.id == "gc":
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_CUDNN_ATTRS:
            if (
                isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "cudnn"
            ):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_BACKEND_TOGGLE_ATTRS:
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_GRAD_TOGGLE_CALLS:
                line = getattr(node, "lineno", "?")
                return (
                    False,
                    f"Line {line}: {func.attr}() is forbidden"
                    " (disabling gradients would bypass verification)",
                )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_DIRECT_CALLS:
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: {node.func.id}() is forbidden"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_ATTR_CALLS:
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: .{node.func.attr}() is forbidden"
                if node.func.attr == "compile":
                    if not (
                        isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
                    ):
                        line = getattr(node, "lineno", "?")
                        return (
                            False,
                            f"Line {line}: .compile() is forbidden (only torch.compile is allowed)",
                        )

        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module in _forbidden_modules or alias.name.startswith("importlib"):
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden import"
                for substr in FORBIDDEN_IMPORT_SUBSTRINGS:
                    if substr in alias.name:
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: forbidden import"
                for forbidden_path in _forbidden_dotted_modules:
                    if alias.name == forbidden_path or alias.name.startswith(forbidden_path + "."):
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: forbidden import"

        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: star imports (from ... import *) are forbidden"
            if not node.module:
                continue
            base_module = node.module.split(".")[0]
            if base_module in _forbidden_modules or node.module.startswith("importlib"):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden import"
            if node.module == "torch":
                for alias in node.names:
                    if alias.name in _forbidden_torch_symbol_imports:
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: importing torch.{alias.name} is forbidden"
            if node.module.startswith("torch.backends"):
                for alias in node.names:
                    if alias.name in _forbidden_torch_backend_symbol_imports:
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: importing torch backend toggle is forbidden"
            for substr in FORBIDDEN_IMPORT_SUBSTRINGS:
                if substr in node.module:
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden import"
            for forbidden_path in _forbidden_dotted_modules:
                if node.module == forbidden_path or node.module.startswith(forbidden_path + "."):
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden import"
            for alias in node.names:
                for substr in FORBIDDEN_IMPORT_SUBSTRINGS:
                    if substr in alias.name:
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: forbidden import"
                full_path = f"{node.module}.{alias.name}"
                for forbidden_path in _forbidden_dotted_modules:
                    if full_path == forbidden_path or full_path.startswith(forbidden_path + "."):
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: forbidden import"
                if full_path in ALLOWED_TORCH_SUBMODULE_IMPORTS:
                    local_name = alias.asname or alias.name
                    torch_submodule_aliases.add(local_name)

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _forbidden_builtins:
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: {node.func.id}() is forbidden"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _forbidden_builtins:
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: .{node.func.attr}() is forbidden"

        # Block torch.load (uses pickle internally — bypasses pickle import ban),
        # including aliased torch names (e.g. tl.load(...)).
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in torch_aliases
        ):
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: torch.load() is forbidden (uses pickle internally)"

        # Block numpy.ctypeslib access (grants access to blocked ctypes via numpy)
        if isinstance(node, ast.Attribute) and node.attr == "ctypeslib":
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: ctypeslib access is forbidden"

        if isinstance(node, ast.Name) and node.id == "__builtins__":
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: __builtins__ access is forbidden"
        if isinstance(node, ast.Attribute) and node.attr == "__builtins__":
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: __builtins__ access is forbidden"

        if isinstance(node, ast.Attribute) and node.attr == "modules":
            if isinstance(node.value, ast.Name) and node.value.id in FORBIDDEN_SYS_MODULE_NAMES:
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr == "__dict__":
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: __dict__ access is forbidden"

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_INTROSPECTION_ATTRS:
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr == "optimizer":
            if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: accessing .optimizer attribute is forbidden"

        # Block dangerous builtins used as decorators (e.g. @property, @staticmethod)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for deco in node.decorator_list:
                if isinstance(deco, ast.Name) and deco.id in _forbidden_names:
                    line = getattr(deco, "lineno", "?")
                    return False, f"Line {line}: decorator @{deco.id} is forbidden"

        # Reject large bytes/bytearray literals regardless of usage.
        # Covers both list and tuple constructors: bytes([...]), bytes((...)),
        # bytearray([...]), bytearray((...)).
        # Attackers embed binary payloads (pickle, ZIP) in raw byte arrays
        # that bypass string-based forbidden-pattern scanning.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("bytes", "bytearray")
            and node.args
            and len(node.args) == 1
            and isinstance(node.args[0], (ast.List, ast.Tuple))
            and len(node.args[0].elts) > _MAX_BYTES_LITERAL_ELTS
        ):
            line = getattr(node, "lineno", "?")
            return False, (
                f"Line {line}: bytes/bytearray literal with {len(node.args[0].elts)} "
                f"elements exceeds limit of {_MAX_BYTES_LITERAL_ELTS}"
            )

        # Block weights_only=False in any .load() call — prevents pickle RCE
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
        ):
            for kw in node.keywords:
                if (
                    kw.arg == "weights_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                ):
                    line = getattr(node, "lineno", "?")
                    return False, (
                        f"Line {line}: weights_only=False is forbidden"
                        " (enables arbitrary code execution via pickle)"
                    )

    return True, None


_FORBIDDEN_STRINGS = FORBIDDEN_STRINGS


def _is_main_guard(node: ast.AST) -> bool:
    """Check if an AST node is an `if __name__ == "__main__":` block.

    Handles both `__name__ == "__main__"` and `"__main__" == __name__`
    for consistency with _collect_main_guard_nodes.
    """
    if not (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], (ast.Eq, ast.Is))
        and len(node.test.comparators) == 1
    ):
        return False
    left, right = node.test.left, node.test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def _validate_code_structure(code: str) -> tuple[bool, str | None]:
    """Validate that train.py has the required structure."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error at line {exc.lineno}: {exc.msg}"

    # Scan the FULL tree — do NOT strip `if __name__ == "__main__":` blocks.
    # Attackers can force them to execute via `__name__: str = "__main__"`
    # (AnnAssign) or other reassignment tricks, so they must be scanned.
    safe, danger_error = _scan_for_dangerous_patterns(tree)
    if not safe:
        return False, f"Security violation: {danger_error}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for pattern in _FORBIDDEN_STRINGS:
                if pattern in node.value:
                    line = getattr(node, "lineno", "?")
                    return False, (
                        f"Security violation: Line {line}: forbidden string pattern detected"
                    )

    # String obfuscation detection — uses shared helpers from security_defs
    _obfuscation_resolvers = [
        ("bytes decode", try_decode_bytes_node),
        ("str(bytes(...))", try_decode_str_bytes_constructor),
        ("str.join()", try_resolve_join),
        ("concatenation", try_resolve_concat),
        ("%-format", try_resolve_format),
        ("f-string", try_resolve_fstring),
    ]
    for node in ast.walk(tree):
        for label, resolver in _obfuscation_resolvers:
            try:
                resolved = resolver(node)
            except SuspiciousConstructionError as exc:
                line = getattr(node, "lineno", "?")
                return False, (f"Security violation: Line {line}: {exc}")
            if resolved is not None:
                for pattern in _FORBIDDEN_STRINGS:
                    if pattern in resolved:
                        line = getattr(node, "lineno", "?")
                        return False, (
                            f"Security violation: Line {line}: forbidden string constructed via {label}"
                        )

    # Scan attribute names (e.g. _e._perf_counter) — AST string scan only
    # catches ast.Constant values, not ast.Attribute.attr names
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            for pattern in _FORBIDDEN_STRINGS:
                if node.attr == pattern:
                    line = getattr(node, "lineno", "?")
                    return False, (
                        f"Security violation: Line {line}: forbidden attribute name '{node.attr}'"
                    )

    inner_steps_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
            inner_steps_found = True
            args = node.args
            if len(args.args) < 5:
                return False, f"inner_steps has {len(args.args)} args, expected at least 5"
            break

    if not inner_steps_found:
        return False, "Missing required function: inner_steps"

    return True, None


def _validate_return_type(result) -> tuple[bool, str | None, InnerStepsResult | None]:
    """Validate that inner_steps returned correct type.

    Rejects proxy/lazy objects that override __getattr__ to defer computation,
    which is a known exploit vector for faking MFU.
    """
    if isinstance(result, InnerStepsResult):
        return True, None, result

    result_type = type(result)
    if any("__getattr__" in cls.__dict__ for cls in result_type.__mro__):
        return (
            False,
            f"Rejected proxy/lazy return type {result_type.__name__}: "
            f"__getattr__ override detected (deferred computation not allowed)",
            None,
        )

    if all(hasattr(result, attr) for attr in ("final_logits", "total_tokens", "final_loss")):
        return (
            True,
            None,
            InnerStepsResult(
                final_logits=result.final_logits,
                total_tokens=result.total_tokens,
                final_loss=result.final_loss,
            ),
        )

    return False, f"Invalid return type from inner_steps: {type(result)}", None


def _load_model(model_path: str, use_random_init: bool = False):
    """Load model from HuggingFace.

    Args:
        model_path: HuggingFace model name/path
        use_random_init: If True, initialize with random weights

    Note:
        Docker evaluation runs with --network=none. Models must be
        pre-cached in the Docker image. If cache is missing, rebuild:
        docker build --network=host -f environments/templar/Dockerfile -t templar-eval:latest .
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if use_random_init:
        # Random initialization
        logger.info(f"Loading model config from {model_path} with RANDOM initialization")

        # Use local cache only (Docker runs with --network=none)
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        device = f"cuda:{_LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        model = model.to(device)
        logger.info(f"Model loaded on {device} (rank {_LOCAL_RANK})")
    else:
        logger.info(f"Loading pretrained model from {model_path}")
        # Use local cache only (Docker runs with --network=none)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )

    model.train()

    # Ensure ALL parameters have requires_grad=True for training
    for param in model.parameters():
        param.requires_grad = True

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def _count_model_params(model: torch.nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def _calculate_mfu(
    total_unique_tokens: int,
    wall_time: float,
    model_params: int,
    gpu_peak_tflops: float = 312.0,
    num_gpus: int = 1,
) -> float:
    """Calculate system-level Model FLOPs Utilization (MFU).

    MFU = total_system_flops / total_system_peak_flops

    The caller provides *total_unique_tokens* — the number of distinct tokens
    processed across the entire system during training:

        total_unique_tokens = tokens_per_rank * dp_size

    where dp_size is the data-parallel dimension of the miner's topology
    (dp_size=num_gpus for pure DDP, dp_size=1 for pure TP, mixed for hybrid).

    Total useful FLOPs = 6 * params * total_unique_tokens (forward + backward).
    Total system peak  = peak_per_gpu * num_gpus * wall_time.

    Args:
        total_unique_tokens: Distinct tokens processed by the system
        wall_time: Wall clock time in seconds
        model_params: Number of model parameters
        gpu_peak_tflops: Per-GPU theoretical peak TFLOPS (A100 80GB = 312 bfloat16)
        num_gpus: Number of GPUs used in evaluation

    Returns:
        MFU as a percentage (0-100)
    """
    if wall_time <= 0 or gpu_peak_tflops <= 0 or num_gpus <= 0:
        return 0.0

    total_system_flops = 6 * model_params * total_unique_tokens
    total_system_peak = gpu_peak_tflops * num_gpus * 1e12 * wall_time

    mfu = (total_system_flops / total_system_peak) * 100

    return min(mfu, 100.0)


def _verify_trainable_params(
    model: torch.nn.Module,
    min_trainable_ratio: float = 1.0,
) -> tuple[bool, str | None, dict]:
    """Check that minimum % of params are trainable (prevents layer freezing)."""
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0

    details = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "min_required": min_trainable_ratio,
    }

    if trainable_ratio < min_trainable_ratio:
        error = (
            f"Insufficient trainable params: {trainable_ratio:.1%} "
            f"({trainable_params:,}/{total_params:,}) - minimum {min_trainable_ratio:.0%} required"
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details

    logger.info(
        f"[PASSED] Trainable params check ({trainable_ratio:.1%} >= {min_trainable_ratio:.0%})"
    )
    return True, None, details


def _verify_params_changed(
    trained_state: dict,
    initial_state: dict,
    min_changed_ratio: float = 0.5,
    element_threshold: float = 1e-6,
) -> tuple[bool, str | None, dict]:
    """Verify that minimum % of individual parameter elements changed during training.

    Operates purely on CPU state dicts -- no model object or GPU memory needed.

    Args:
        trained_state: State dict after training (CPU tensors)
        initial_state: State dict before training (CPU tensors)
        min_changed_ratio: Minimum fraction of elements that must change
        element_threshold: Minimum absolute change for an element to count as "changed"
    """
    total_elements = 0
    changed_elements = 0

    for name, trained_val in trained_state.items():
        if name not in initial_state:
            continue
        initial_val = initial_state[name]
        if trained_val.shape != initial_val.shape:
            continue
        element_diffs = (trained_val.cpu().float() - initial_val.cpu().float()).abs()
        changed_mask = element_diffs > element_threshold

        total_elements += trained_val.numel()
        changed_elements += changed_mask.sum().item()

    changed_ratio = changed_elements / total_elements if total_elements > 0 else 0.0

    details = {
        "total_elements": total_elements,
        "changed_elements": int(changed_elements),
        "changed_ratio": changed_ratio,
        "min_required": min_changed_ratio,
        "element_threshold": element_threshold,
    }

    if changed_ratio < min_changed_ratio:
        error = (
            f"Insufficient parameter updates: {changed_ratio:.1%} "
            f"({int(changed_elements):,}/{total_elements:,} elements) - minimum {min_changed_ratio:.0%} required"
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details

    logger.info(
        f"[PASSED] Parameter changes check ({changed_ratio:.1%} >= {min_changed_ratio:.0%})"
    )
    return True, None, details


def _get_cached_model(model_path: str, use_random_init: bool = False):
    """Get model from cache or load it."""
    cached = _CACHE.get("model")
    cached_path = _CACHE.get("model_path")
    cached_random_init = _CACHE.get("use_random_init")

    # Cache hit only if path AND init mode match
    if cached is not None and cached_path == model_path and cached_random_init == use_random_init:
        if _CACHE.get("initial_state"):
            cached.load_state_dict(_CACHE["initial_state"])
            cached.to(torch.device(f"cuda:{_LOCAL_RANK}"))
        return cached

    model = _load_model(model_path, use_random_init=use_random_init)
    _CACHE["model"] = model
    _CACHE["model_path"] = model_path
    _CACHE["use_random_init"] = use_random_init
    _CACHE["initial_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model


def _create_data_iterator(
    data: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    *,
    offset: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[torch.Tensor]:
    """Create infinite data iterator with optional DDP-aware sharding.

    When ``world_size > 1`` each rank receives non-overlapping batches:
    rank *k* starts at ``offset + k * batch_size`` and advances by
    ``world_size * batch_size`` per step.
    """
    if data.size(1) < sequence_length:
        raise ValueError(f"Data sequence length {data.size(1)} < required {sequence_length}")

    data = data[:, :sequence_length]
    num_samples = data.size(0)
    stride = batch_size * world_size

    def _iter():
        idx = (offset + rank * batch_size) % num_samples if num_samples > 0 else 0
        while True:
            end_idx = idx + batch_size
            if end_idx > num_samples:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = (idx + stride) % num_samples

    return _iter()


def _create_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create standard AdamW optimizer (fused on CUDA for performance)."""
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        fused=use_fused,
    )


_REFERENCE_MICRO_BATCH_SIZE = 4


def _run_reference(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
    ddp_model: torch.nn.Module | None = None,
) -> InnerStepsResult:
    """Run reference implementation for comparison.

    Uses moderately large micro-batches (8) to balance activation memory
    against floating-point divergence from gradient accumulation.  FSDP
    FULL_SHARD (single-unit wrap, no auto_wrap_policy) stores activations
    for ALL layers; micro_batch=16 OOMs on 3B Qwen (36 layers, 11008
    intermediate).  For batch_size=32, micro_batch=8 means 4 accumulations
    per step — low rounding compared to micro_batch=2 (16 accumulations).

    When *ddp_model* is provided, the forward pass uses the distributed
    wrapper (DDP or FSDP) while gradient capture reads from the unwrapped
    *model*.  Both DDP and FSDP expose ``no_sync()`` for gradient
    accumulation, so micro-batching works unchanged for either wrapper.

    Returns:
        InnerStepsResult with final logits, token count, and loss.
    """
    from contextlib import nullcontext

    _enforce_backend_state()

    eval_model = ddp_model if ddp_model is not None else model
    use_no_sync = hasattr(eval_model, "no_sync")
    mbs = _REFERENCE_MICRO_BATCH_SIZE

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        micro_batches = [batch[i : i + mbs] for i in range(0, batch.size(0), mbs)]
        num_accum = len(micro_batches)

        step_logits = None
        step_loss_sum = 0.0

        for idx, mb in enumerate(micro_batches):
            input_ids = mb[:, :-1]
            labels = mb[:, 1:]

            sync_ctx = (
                eval_model.no_sync() if use_no_sync and idx < num_accum - 1 else nullcontext()
            )
            with sync_ctx:
                outputs = eval_model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = _real_cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                (loss / num_accum).backward()

            step_logits = logits
            step_loss_sum += float(loss.item())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_logits = step_logits.detach().float()
        final_loss = step_loss_sum / num_accum

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


def _verify_final_weights(
    candidate_state: dict,
    reference_final_state: dict,
    max_relative_error: float = 0.05,
) -> tuple[bool, str | None, dict]:
    """Verify miner's final weights match reference after full training.

    Operates purely on CPU state dicts -- no model object or GPU memory needed.
    Compares layer-by-layer to stay memory efficient.

    Args:
        candidate_state: Miner's state dict after training (CPU tensors)
        reference_final_state: Reference state dict after training (CPU tensors)
        max_relative_error: Maximum allowed |w_miner - w_ref| / |w_ref|

    Returns:
        Tuple of (success, error_message, details)
    """
    details: dict = {
        "relative_error_threshold": max_relative_error,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Final model weight comparison")
    logger.info("=" * 60)

    diff_norm_sq = 0.0
    ref_norm_sq = 0.0
    total_elements = 0
    shape_mismatched_layers = 0
    error_exceeded_layers = 0

    for name, cand_val in candidate_state.items():
        if name not in reference_final_state:
            continue

        ref_param = reference_final_state[name]
        if cand_val.shape != ref_param.shape:
            shape_mismatched_layers += 1
            continue
        diff = cand_val.cpu().float() - ref_param.cpu().float()

        layer_diff_sq = (diff * diff).sum().item()
        layer_ref_sq = (ref_param.cpu().float() * ref_param.cpu().float()).sum().item()

        diff_norm_sq += layer_diff_sq
        ref_norm_sq += layer_ref_sq
        total_elements += cand_val.numel()

        if layer_ref_sq > 0:
            layer_rel_error = (layer_diff_sq**0.5) / (layer_ref_sq**0.5)
            if not math.isfinite(layer_rel_error) or layer_rel_error > max_relative_error:
                error_exceeded_layers += 1

    mismatched_layers = shape_mismatched_layers + error_exceeded_layers

    ref_norm = ref_norm_sq**0.5
    diff_norm = diff_norm_sq**0.5

    if ref_norm > 0:
        relative_error = diff_norm / ref_norm
    else:
        relative_error = 0.0 if diff_norm == 0 else float("inf")

    details["relative_error"] = relative_error
    details["diff_norm"] = diff_norm
    details["ref_norm"] = ref_norm
    details["total_elements"] = total_elements

    logger.info(f"[CHECK] Weight relative error: {relative_error:.6f}")
    logger.info(f"   |w_miner - w_ref|: {diff_norm:.6f}")
    logger.info(f"   |w_ref|: {ref_norm:.6f}")
    logger.info(f"   Max allowed: {max_relative_error:.6f}")
    logger.info(f"   Total elements: {total_elements:,}")
    logger.info(f"   Shape mismatched layers: {shape_mismatched_layers}")
    logger.info(f"   Per-layer error exceeded layers: {error_exceeded_layers}")
    details["mismatched_layers"] = mismatched_layers
    details["shape_mismatched_layers"] = shape_mismatched_layers
    details["error_exceeded_layers"] = error_exceeded_layers

    if shape_mismatched_layers > 0:
        error = (
            f"{shape_mismatched_layers} layer(s) have shape mismatches — "
            f"model architecture was modified"
        )
        details["checks_failed"].append({"check": "shape_mismatch", "error": error})
        details["error_code"] = "shape_mismatch"
        logger.error(f"[FAILED] {error}")
        return False, error, details

    # Guard against NaN injection in weights
    if not math.isfinite(relative_error):
        error = (
            f"Weight relative error is non-finite ({relative_error}) - "
            "possible NaN injection detected"
        )
        details["checks_failed"].append({"check": "weight_nan_guard", "error": error})
        details["error_code"] = "weight_nan_injection"
        logger.error(f"[FAILED] {error}")
        return False, error, details

    if relative_error > max_relative_error:
        error = (
            f"Final weight relative error {relative_error:.6f} exceeds threshold "
            f"{max_relative_error:.6f} (|w_miner - w_ref| / |w_ref|)"
        )
        details["checks_failed"].append({"check": "weight_relative_error", "error": error})
        details["error_code"] = "weight_mismatch"
        logger.error(f"[FAILED] {error}")
        return False, error, details

    details["checks_passed"].append("weight_relative_error")
    logger.info("[PASSED] Final model weights match reference")

    logger.info("=" * 60)
    logger.info("VERIFICATION: WEIGHT CHECK PASSED")
    logger.info("=" * 60)

    return True, None, details


def _verify_outputs(
    reference: InnerStepsResult,
    candidate: InnerStepsResult,
    expected_tokens: int,
    reference_final_state: dict | None = None,
    candidate_final_state: dict | None = None,
    max_loss_difference: float = 0.3,
    weight_relative_error_max: float = 0.008,
) -> tuple[bool, str | None, dict]:
    """Verify candidate outputs match reference.

    Verification checks (uniform across all parallelism strategies):
    1. Token count matches expected
    2. Loss is valid and similar to reference (small difference)
    3. Final weight verification (captures optimizer step correctness)

    Args:
        reference: Reference implementation results
        candidate: Miner's implementation results
        expected_tokens: Expected token count
        reference_final_state: Reference state dict after full training (CPU)
        candidate_final_state: Miner's state dict after training (CPU)
        max_loss_difference: Maximum allowed |candidate_loss - reference_loss|
        weight_relative_error_max: Max relative error for final weight comparison

    Returns:
        Tuple of (success, error_message, verification_details)
    """
    details = {
        "expected_tokens": expected_tokens,
        "candidate_tokens": candidate.total_tokens,
        "candidate_loss": candidate.final_loss,
        "reference_loss": reference.final_loss if reference else None,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Starting output verification")
    logger.info("=" * 60)

    # 1. Verify token count matches expected
    logger.info(
        f"[CHECK 1/3] Token count: expected={expected_tokens}, got={candidate.total_tokens}"
    )
    if candidate.total_tokens != expected_tokens:
        error = f"Token count mismatch: expected {expected_tokens}, got {candidate.total_tokens}"
        details["checks_failed"].append({"check": "token_count", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    details["checks_passed"].append("token_count")
    logger.info("[PASSED] Token count matches")

    # 2. Verify loss is reasonable and similar to reference
    logger.info(f"[CHECK 2/3] Loss validity: candidate_loss={candidate.final_loss:.6f}")
    if candidate.final_loss != candidate.final_loss:  # NaN check
        error = "Loss is NaN"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if candidate.final_loss <= 0:
        error = f"Loss must be positive (cross-entropy > 0): got {candidate.final_loss:.4f}"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if candidate.final_loss > 100:
        error = f"Loss unreasonable: {candidate.final_loss:.4f} (expected 1-10)"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    # Compare losses - check absolute difference
    if reference is None:
        error = "No reference result available for loss comparison"
        details["checks_failed"].append({"check": "loss_comparison", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    loss_difference = abs(candidate.final_loss - reference.final_loss)
    details["loss_difference"] = loss_difference
    details["reference_loss"] = reference.final_loss
    logger.info(
        f"   Reference loss: {reference.final_loss:.4f}, Candidate loss: {candidate.final_loss:.4f}"
    )
    logger.info(f"   Loss difference: {loss_difference:.4f} (max allowed: {max_loss_difference})")

    if loss_difference > max_loss_difference:
        error = (
            f"Loss difference too large: candidate={candidate.final_loss:.4f}, "
            f"reference={reference.final_loss:.4f}, diff={loss_difference:.4f} > {max_loss_difference}"
        )
        details["checks_failed"].append({"check": "loss_comparison", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    details["checks_passed"].append("loss_validity")
    logger.info("[PASSED] Loss is valid and similar to reference")

    # 3. Final weight verification (verifies optimizer step correctness)
    if reference_final_state is not None and candidate_final_state is not None:
        logger.info("[CHECK 3/3] Final weight verification")
        weight_ok, weight_error, weight_details = _verify_final_weights(
            candidate_final_state,
            reference_final_state,
            max_relative_error=weight_relative_error_max,
        )
        details["weight_verification"] = weight_details

        if not weight_ok:
            details["checks_failed"].append({"check": "weight_verification", "error": weight_error})
            return False, weight_error, details
        details["checks_passed"].append("weight_verification")
    else:
        logger.warning("Skipping weight verification (no reference state available)")

    logger.info("=" * 60)
    logger.info("VERIFICATION: ALL CHECKS PASSED")
    logger.info(f"   Checks passed: {details['checks_passed']}")
    logger.info("=" * 60)

    return True, None, details


class Actor:
    """Templar MFU Evaluation Actor for Affinetes (Validator-Owned).

    This Actor is owned by the validator, not the miner.
    Code is passed directly from the validator (downloaded from URL).
    """

    async def evaluate(
        self,
        task_id: int = 0,
        seed: str = "0:0:0",
        model_url: str = "",
        data_url: str = "",
        steps: int = 5,
        batch_size: int = 8,
        timeout: int = 600,
        sequence_length: int | None = None,
        data_samples: int = 10000,
        code: str = "",  # Miner's code passed directly
        # Verification settings
        max_loss_difference: float = 0.3,
        use_random_init: bool = True,
        min_trainable_params_ratio: float = 1.0,
        min_params_changed_ratio: float = 0.75,
        # Weight verification (0.8% — micro-batch=8 reference keeps FP drift low)
        weight_relative_error_max: float = 0.008,
        # Timer integrity
        timer_divergence_threshold: float = 0.005,
        # MFU calculation
        gpu_peak_tflops: float = 312.0,
        max_plausible_mfu: float = 75.0,
        min_mfu: float = 50.0,
        require_cuda_timing: bool = True,
        model_params_override: int | None = None,
        num_gpus: int = 1,
    ) -> dict:
        """
        Run MFU evaluation on miner's code.

        Args:
            task_id: Evaluation run identifier
            seed: Deterministic seed string (format: "block:uid:run_idx")
            model_url: HuggingFace model name
            data_url: HuggingFace dataset name
            steps: Number of training steps
            batch_size: Batch size
            timeout: Maximum seconds
            sequence_length: Sequence length
            data_samples: Number of data samples
            code: Miner's train.py code (passed directly from validator)
            max_loss_difference: Max allowed |candidate_loss - reference_loss|
            use_random_init: Use random weights
            min_trainable_params_ratio: Min % params that must be trainable
            min_params_changed_ratio: Min % params that must change
            weight_relative_error_max: Max relative error for final weight check (e.g., 0.008 = 0.8%)
            timer_divergence_threshold: Max divergence between timer sources (e.g., 0.005 = 0.5%)
            gpu_peak_tflops: GPU peak TFLOPS for MFU calculation
            min_mfu: Minimum MFU threshold — submissions below this are rejected
            model_params_override: Override model param count (None = auto-detect)

        Returns:
            Dict with: mfu, tps, total_tokens, wall_time_seconds, success, error, code
        """
        # Validate code
        if not code:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "No code provided",
                "seed": seed,
            }

        # Validate model/data
        if not model_url or not data_url:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Missing model_url or data_url",
                "seed": seed,
            }

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        seed_value = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2**32)
        _set_deterministic(seed_value)

        # Validate code structure
        code_ok, code_error = _validate_code_structure(code)
        if not code_ok:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": code_error,
                "seed": seed,
                "code": code,
            }

        # Write code to temp file for loading as module
        train_path = CACHE_DIR / "miner_train.py"
        try:
            train_path.write_text(code)
        except Exception as e:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"Failed to write train.py: {e}",
                "seed": seed,
            }

        if time.monotonic() > deadline:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Timeout before evaluation",
                "seed": seed,
            }

        from datetime import timedelta

        import torch.distributed as dist

        _multi_gpu = num_gpus > 1 and _WORLD_SIZE > 1

        # Detect parallelism topology from source (AST only, no execution).
        par_config = _detect_strategy_from_source(code, num_gpus)

        if par_config is None:
            if _multi_gpu:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": (
                        "Multi-GPU evaluation requires get_strategy() in train.py "
                        'returning "ddp"/"fsdp"/"tp" or {"dp_size": N, "tp_size": M}'
                    ),
                    "seed": seed,
                    "code": code,
                }
            par_config = ParallelismConfig(dp_size=1, tp_size=1)

        if par_config.dp_size * par_config.tp_size != num_gpus:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": (
                    f"dp_size({par_config.dp_size}) * tp_size({par_config.tp_size}) "
                    f"= {par_config.dp_size * par_config.tp_size} != num_gpus({num_gpus})"
                ),
                "seed": seed,
                "code": code,
            }

        logger.info(
            f"Miner parallelism: dp_size={par_config.dp_size}, tp_size={par_config.tp_size}"
        )

        data_rank = _LOCAL_RANK // par_config.tp_size if _multi_gpu else 0
        data_world = par_config.dp_size if _multi_gpu else 1

        try:
            # ---------------------------------------------------------------
            # CRITICAL: Run reference BEFORE loading miner module.
            # Miner module-level code executes on import and can monkey-patch
            # torch functions (e.g. F.cross_entropy = fake).  If we load the
            # miner first, the reference would use the patched functions and
            # produce the same wrong results as the miner, making all
            # verification checks (loss, gradient, weight) pass on garbage.
            # ---------------------------------------------------------------

            # Load model and data
            _log_vram("before-model-load")
            model = _get_cached_model(model_url, use_random_init=use_random_init)
            model_params = model_params_override or _count_model_params(model)
            logger.info(f"Model loaded: {model_params:,} parameters, random_init={use_random_init}")

            data = _load_hf_dataset(
                dataset_name=data_url,
                model_name=model_url,
                num_samples=data_samples,
                sequence_length=sequence_length or EVAL_SEQUENCE_LENGTH,
                validator_seed=seed,  # Unpredictable sampling
            )
            device = torch.device(f"cuda:{_LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
            if _multi_gpu:
                if not dist.is_initialized():
                    torch.cuda.set_device(_LOCAL_RANK)
                    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=3600))
                    logger.info(f"Initialized process group: rank {_LOCAL_RANK}/{_WORLD_SIZE}")

            seq_len = sequence_length or EVAL_SEQUENCE_LENGTH

            # Cache initial state on all ranks (needed to restore after reference)
            initial_state = _CACHE.get("initial_state")
            if initial_state is None:
                initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                _CACHE["initial_state"] = initial_state

            # ── Unified reference baseline ──────────────────────────────────
            # Multi-GPU: always FSDP FULL_SHARD (scales to any model size,
            # all ranks participate regardless of miner strategy).
            # Data distribution matches the miner's declared topology:
            # ranks in the same TP group share data, different DP groups
            # get different data.  dp_size/tp_size drive the formula.
            reference: InnerStepsResult | None = None
            reference_final_state: dict | None = None
            ref_fsdp: torch.nn.Module | None = None

            data_iter_ref = _create_data_iterator(
                data,
                batch_size,
                seq_len,
                rank=data_rank,
                world_size=data_world,
            )

            if _multi_gpu:
                from torch.distributed.fsdp import (
                    FullStateDictConfig,
                    ShardingStrategy,
                    StateDictType,
                )
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,  # noqa: N817
                )

                # Gradient checkpointing is incompatible with FULL_SHARD when
                # the model is wrapped as a single FSDP unit (no auto_wrap_policy):
                # FSDP frees param storage after forward, but checkpointing tries
                # to re-access it during backward.  Disable it for the reference.
                if hasattr(model, "gradient_checkpointing_disable"):
                    model.gradient_checkpointing_disable()

                ref_fsdp = FSDP(
                    model,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    device_id=torch.device(f"cuda:{_LOCAL_RANK}"),
                )
                optimizer_ref = _create_optimizer(ref_fsdp)

                reference = _run_reference(
                    model,
                    data_iter_ref,
                    optimizer_ref,
                    steps,
                    device,
                    ddp_model=ref_fsdp,
                )

                # All ranks must call state_dict (FSDP all-gathers internally);
                # rank0_only=True means only rank 0 receives the full tensors.
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(ref_fsdp, StateDictType.FULL_STATE_DICT, save_policy):
                    full_sd = ref_fsdp.state_dict()
                if _IS_RANK_0:
                    reference_final_state = {k: v.detach().clone() for k, v in full_sd.items()}
                    logger.info("Captured FSDP reference final state for weight verification")
                del full_sd
                _log_vram("after-reference-run")

                if reference is not None and reference.final_logits is not None:
                    reference.final_logits = reference.final_logits.cpu()

                del ref_fsdp
                ref_fsdp = None
                del optimizer_ref
                del data_iter_ref
                _CACHE.pop("model", None)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                model = _load_model(model_url, use_random_init=use_random_init)
                _CACHE["model"] = model
                model.load_state_dict(initial_state)
                model.to(torch.device(f"cuda:{_LOCAL_RANK}"))
                dist.barrier()
            else:
                optimizer_ref = _create_optimizer(model)

                reference = _run_reference(
                    model,
                    data_iter_ref,
                    optimizer_ref,
                    steps,
                    device,
                )

                reference_final_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                logger.info("Captured single-GPU reference final state for weight verification")
                _log_vram("after-reference-run")

                if reference is not None and reference.final_logits is not None:
                    reference.final_logits = reference.final_logits.cpu()

                del optimizer_ref
                del data_iter_ref
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model.load_state_dict(initial_state)
                model.to(torch.device(f"cuda:{_LOCAL_RANK}"))

            # Capture vault timer refs into locals BEFORE loading miner code.
            # Closure variables inside the vault are immune to frame.f_globals
            # patching, and these locals can't be reached by the miner either.
            _vpc, _vmo, _vsy, _vet, _vce = _vault_get_timers()
            _vpc_id, _vmo_id, _vsy_id, _vet_id, _vce_id = _vault_get_real_ids()

            # NOW load the miner module — after the reference is safely captured.
            # Module-level side-effects (monkey-patches etc.) can no longer affect
            # the reference results.
            miner_module = _load_miner_module(train_path)

            if not hasattr(miner_module, "inner_steps"):
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": "train.py missing inner_steps function",
                    "seed": seed,
                    "code": code,
                }

            # Verify no critical torch functions were tampered with during
            # module loading.  This catches F.cross_entropy = fake, etc.
            if id(F.cross_entropy) != _REAL_CE_ID:
                logger.error(
                    "F.cross_entropy tampered after miner module load! "
                    f"Expected id {_REAL_CE_ID}, got {id(F.cross_entropy)}"
                )
                F.cross_entropy = _real_cross_entropy
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": "torch.nn.functional tampering detected after module load",
                    "error_code": "functional_tampering",
                    "seed": seed,
                    "code": code,
                }

            # Warmup uses randomized weights and a random data offset.
            # The real initial_state is restored after warmup completes.
            with torch.no_grad():
                for p in model.parameters():
                    p.uniform_(-0.1, 0.1)

            warmup_data_offset = int.from_bytes(os.urandom(4), "big") % max(data.size(0), 1)
            logger.info(
                "Warmup using perturbed weights and data offset=%d",
                warmup_data_offset,
            )

            warmup_steps = 2
            data_iter_warmup = _create_data_iterator(
                data,
                batch_size,
                seq_len,
                offset=warmup_data_offset,
                rank=data_rank,
                world_size=data_world,
            )
            optimizer_warmup = None

            logger.info(f"Running {warmup_steps} warmup step(s) to check for basic errors...")
            warmup_failure: dict | None = None
            try:
                # Hide env modules during miner execution to prevent timer tampering.
                with _hide_sensitive_env_modules():
                    warmup_result = miner_module.inner_steps(
                        model=model,
                        data_iterator=data_iter_warmup,
                        optimizer=optimizer_warmup,
                        num_steps=warmup_steps,
                        device=device,
                        num_gpus=num_gpus,
                    )

                # Quick validation of warmup result (only rank 0 validates)
                if _IS_RANK_0:
                    if warmup_result is None:
                        warmup_failure = {
                            "task_id": task_id,
                            "mfu": 0.0,
                            "tps": 0.0,
                            "total_tokens": 0,
                            "wall_time_seconds": 0.0,
                            "success": False,
                            "error": "inner_steps returned None (warmup check)",
                            "seed": seed,
                            "code": code,
                        }
                    elif not hasattr(warmup_result, "final_logits"):
                        warmup_failure = {
                            "task_id": task_id,
                            "mfu": 0.0,
                            "tps": 0.0,
                            "total_tokens": 0,
                            "wall_time_seconds": 0.0,
                            "success": False,
                            "error": "inner_steps result missing 'final_logits' attribute",
                            "seed": seed,
                            "code": code,
                        }
                    else:
                        logits = warmup_result.final_logits
                        if logits is None:
                            warmup_failure = {
                                "task_id": task_id,
                                "mfu": 0.0,
                                "tps": 0.0,
                                "total_tokens": 0,
                                "wall_time_seconds": 0.0,
                                "success": False,
                                "error": "final_logits is None - must return actual logits for verification",
                                "error_code": "missing_logits",
                                "seed": seed,
                                "code": code,
                            }
                        elif len(logits.shape) != 3:
                            warmup_failure = {
                                "task_id": task_id,
                                "mfu": 0.0,
                                "tps": 0.0,
                                "total_tokens": 0,
                                "wall_time_seconds": 0.0,
                                "success": False,
                                "error": f"Logits shape mismatch: expected 3D (batch, seq, vocab), got {logits.shape}",
                                "error_code": "invalid_logits_shape",
                                "seed": seed,
                                "code": code,
                            }

                    if warmup_failure is None:
                        logger.info("Warmup passed - proceeding with verification checks")
                        trainable_ok, trainable_error, trainable_details = _verify_trainable_params(
                            model, min_trainable_params_ratio
                        )
                        if not trainable_ok:
                            warmup_failure = {
                                "task_id": task_id,
                                "mfu": 0.0,
                                "tps": 0.0,
                                "total_tokens": 0,
                                "wall_time_seconds": 0.0,
                                "success": False,
                                "error": trainable_error,
                                "error_code": "insufficient_trainable_params",
                                "seed": seed,
                                "code": code,
                                "diagnostics": {"trainable_params": trainable_details},
                            }

            except Exception as e:
                error_msg = f"Early termination (warmup failed): {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                warmup_failure = {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": error_msg,
                    "seed": seed,
                    "code": code,
                }

            # Synchronize warmup result across all ranks to prevent deadlock.
            if _multi_gpu:
                local_fail = 1 if warmup_failure is not None else 0
                fail_tensor = torch.tensor([local_fail], dtype=torch.int32, device=device)
                dist.all_reduce(fail_tensor, op=dist.ReduceOp.MAX)
                if fail_tensor.item() > 0:
                    if warmup_failure is not None:
                        return warmup_failure
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "Warmup failed on another rank",
                        "seed": seed,
                        "code": code,
                    }
            elif warmup_failure is not None:
                return warmup_failure

            # Reset model after warmup. Multi-GPU strategies (DDP/FSDP/TP) wrap
            # the model in-place leaving residual buffers/hooks that prevent full
            # VRAM reclaim. Reload from scratch to guarantee a clean slate.
            del optimizer_warmup
            del data_iter_warmup
            if _multi_gpu:
                _CACHE.pop("model", None)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = _load_model(model_url, use_random_init=use_random_init)
                _CACHE["model"] = model
                model.load_state_dict(initial_state)
                model.to(torch.device(f"cuda:{_LOCAL_RANK}"))
            else:
                model.load_state_dict(initial_state)
                model.to(torch.device(f"cuda:{_LOCAL_RANK}"))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Full timed evaluation — optimizer is always None so miners
            # create their own (uniform across all parallelism strategies).
            _log_vram("before-timed-eval")
            data_iter_miner = _create_data_iterator(
                data,
                batch_size,
                seq_len,
                rank=data_rank,
                world_size=data_world,
            )
            optimizer_miner = None

            _reset_torch_state()
            _enforce_backend_state()

            _model_config_snapshot = None
            if hasattr(model, "config"):
                _cfg = model.config
                _model_config_snapshot = {
                    k: getattr(_cfg, k) for k in vars(_cfg) if not k.startswith("_")
                }

            # ── Runtime timer integrity check ────────────────────────────
            # Verify module globals against vault-stored IDs (closure-protected,
            # immune to frame.f_globals patching).
            def _runtime_ids_ok() -> bool:
                ok = (
                    id(_perf_counter) == _vpc_id
                    and id(_monotonic) == _vmo_id
                    and id(_cuda_synchronize) == _vsy_id
                    and id(F.cross_entropy) == _vce_id
                )
                if _vet_id is not None:
                    ok = ok and id(torch.cuda.Event.elapsed_time) == _vet_id
                return ok

            if not _runtime_ids_ok():
                logger.error("Runtime references tampered BEFORE timed section!")
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": "Runtime tampering detected (pre-eval)",
                    "error_code": "timer_tampering",
                    "seed": seed,
                    "code": code,
                }

            # Hardware-level timing events. Created BEFORE miner code runs.
            _cuda_start_event = None
            _cuda_end_event = None
            if torch.cuda.is_available():
                _cuda_start_event = torch.cuda.Event(enable_timing=True)
                _cuda_end_event = torch.cuda.Event(enable_timing=True)

            # Use vault-sourced local refs — captured before miner module load,
            # immune to frame.f_globals patching.
            _pc = _vpc
            _mo = _vmo
            _sy = _vsy

            if _multi_gpu:
                dist.barrier()

            _sy()
            start_perf = _pc()
            start_mono = _mo()
            if _cuda_start_event is not None:
                _cuda_start_event.record()

            # Hide env modules during timed miner execution to prevent timer tampering.
            with _hide_sensitive_env_modules():
                miner_result = miner_module.inner_steps(
                    model=model,
                    data_iterator=data_iter_miner,
                    optimizer=optimizer_miner,
                    num_steps=steps,
                    device=device,
                    num_gpus=num_gpus,
                )

            # Anti-deferred-computation: force materialization of training
            # result fields BEFORE stopping the timer.  A lazy proxy (e.g.
            # one using __getattr__ to defer real training until field access)
            # would otherwise run its heavy work after timing ends, faking
            # high MFU.
            #
            # We deliberately EXCLUDE final_state from the timed section:
            # for FSDP miners, model.state_dict() triggers a full_state_dict
            # all-gather across ranks — that's verification overhead, not
            # training compute.  Including it would unfairly penalize FSDP.
            #
            # This is safe because final_logits + final_loss + CUDA sync
            # already prove real GPU training happened (the loss must match
            # the seed-specific reference within 0.3, which is unpredictable
            # without actually running the forward/backward passes).
            _mat_logits = getattr(miner_result, "final_logits", None)
            _mat_tokens = getattr(miner_result, "total_tokens", None)
            _mat_loss = getattr(miner_result, "final_loss", None)
            if isinstance(_mat_logits, torch.Tensor):
                _ = _mat_logits.shape
            _sy()

            if _cuda_end_event is not None:
                _cuda_end_event.record()
            _sy()
            end_perf = _pc()
            end_mono = _mo()

            # Access final_state OUTSIDE the timer (verification-only).
            # Guard against lazy values by checking each tensor is real.
            _mat_state = getattr(miner_result, "final_state", None)
            if isinstance(_mat_state, dict):
                for _sk, _sv in _mat_state.items():
                    if isinstance(_sv, torch.Tensor):
                        _sv.data_ptr()
                    elif _sv is not None and not isinstance(_sv, (int, float, bool, str)):
                        _state_vtype = type(_sv)
                        if any(
                            "__getattr__" in cls.__dict__ or "__get__" in cls.__dict__
                            for cls in _state_vtype.__mro__
                        ):
                            logger.error(
                                f"Lazy/proxy object detected in final_state['{_sk}']: "
                                f"{_state_vtype.__name__}"
                            )
                            _mat_state = None
                            break

            # Post-eval tamper check on module-level refs
            if not _runtime_ids_ok():
                logger.error("Runtime references tampered DURING timed section!")
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": "Runtime tampering detected (post-eval)",
                    "error_code": "timer_tampering",
                    "seed": seed,
                    "code": code,
                }

            wall_time_perf = end_perf - start_perf
            wall_time_mono = end_mono - start_mono

            # CUDA event wall time — call via vault-sourced ref so a
            # monkey-patched Event.elapsed_time cannot affect the result.
            cuda_wall_time = None
            if _cuda_start_event is not None and _cuda_end_event is not None:
                _cuda_end_event.synchronize()
                cuda_wall_time = _vet(_cuda_start_event, _cuda_end_event) / 1000.0

            # Three-way timer cross-check: perf_counter vs monotonic vs CUDA events
            def _timer_divergence(a: float, b: float) -> float:
                return abs(a - b) / max(b, 1e-9)

            _timer_pairs = [
                ("perf_counter", wall_time_perf, "monotonic", wall_time_mono),
            ]
            if cuda_wall_time is not None and cuda_wall_time > 0:
                _timer_pairs.append(("perf_counter", wall_time_perf, "cuda_events", cuda_wall_time))
                _timer_pairs.append(("monotonic", wall_time_mono, "cuda_events", cuda_wall_time))

            for name_a, val_a, name_b, val_b in _timer_pairs:
                if val_b > 0 and _timer_divergence(val_a, val_b) > timer_divergence_threshold:
                    logger.warning(
                        f"Timer integrity check FAILED: {name_a}={val_a:.4f}s "
                        f"vs {name_b}={val_b:.4f}s "
                        f"(divergence={_timer_divergence(val_a, val_b) * 100:.1f}%)"
                    )
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": cuda_wall_time or wall_time_mono,
                        "success": False,
                        "error": f"Timer integrity violation: {name_a} vs {name_b}",
                        "error_code": "timer_tampering",
                        "seed": seed,
                        "code": code,
                    }

            # Require CUDA event timing for wall time measurement.
            if require_cuda_timing and (cuda_wall_time is None or cuda_wall_time <= 0):
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time_mono,
                    "success": False,
                    "error": "CUDA event timing unavailable; rejecting evaluation",
                    "error_code": "timer_source_unavailable",
                    "seed": seed,
                    "code": code,
                }

            # Use CUDA event time as canonical source, fall back to monotonic
            wall_time = (
                cuda_wall_time
                if cuda_wall_time is not None and cuda_wall_time > 0
                else wall_time_mono
            )
            logger.info(
                f"Timing: wall_time={wall_time:.2f}s "
                f"(perf_counter={wall_time_perf:.2f}s, monotonic={wall_time_mono:.2f}s"
                f"{f', cuda_events={cuda_wall_time:.2f}s' if cuda_wall_time else ''})"
            )

            # For multi-GPU, use the minimum wall_time across ranks.
            # Rank 0 carries asymmetric overhead from state-dict gathering
            # (FSDP full_state_dict + CPU offload) that is verification cost,
            # not training compute — exclude it from MFU.
            if _multi_gpu:
                _wt = torch.tensor([wall_time], dtype=torch.float64, device=device)
                _wt_max_t = _wt.clone()
                dist.all_reduce(_wt, op=dist.ReduceOp.MIN)
                dist.all_reduce(_wt_max_t, op=dist.ReduceOp.MAX)
                _wt_min = _wt.item()
                _wt_max = _wt_max_t.item()
                _wt_rel_div = (_wt_max - _wt_min) / max(_wt_max, 1e-9)
                if _wt_rel_div > 0.20:
                    logger.error(
                        f"Rank wall_time divergence too high: "
                        f"min={_wt_min:.2f}s max={_wt_max:.2f}s "
                        f"relative_divergence={_wt_rel_div:.1%}"
                    )
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": _wt_max,
                        "success": False,
                        "error": f"Rank wall_time divergence {_wt_rel_div:.1%} exceeds 20% threshold "
                        f"(min={_wt_min:.2f}s, max={_wt_max:.2f}s). "
                        f"Possible asymmetric work distribution.",
                        "error_code": "wall_time_divergence",
                        "seed": seed,
                        "code": code,
                    }
                if wall_time - _wt_min > 0.5:
                    logger.info(
                        f"Using min wall_time across ranks: {_wt_min:.2f}s "
                        f"(rank 0 measured {wall_time:.2f}s, "
                        f"delta={wall_time - _wt_min:.2f}s verification overhead)"
                    )
                    wall_time = _wt_min

            # Non-rank-0: skip verification (barrier in finally block)
            if not _IS_RANK_0:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": True,
                    "error": None,
                    "seed": seed,
                    "code": code,
                }

            _safe_config_changes = {
                "use_cache",
                "output_hidden_states",
                "output_attentions",
                "return_dict",
            }
            if _model_config_snapshot is not None and hasattr(model, "config"):
                _cfg_after = model.config
                _config_changes = []
                for _ck, _cv in _model_config_snapshot.items():
                    if _ck in _safe_config_changes:
                        continue
                    _cv_after = getattr(_cfg_after, _ck, _cv)
                    if _cv_after != _cv:
                        _config_changes.append(f"{_ck}: {_cv!r} -> {_cv_after!r}")
                if _config_changes:
                    logger.error(f"Model config tampered: {_config_changes}")
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": wall_time,
                        "success": False,
                        "error": f"Model config was modified during training: {', '.join(_config_changes[:5])}",
                        "error_code": "config_tampering",
                        "seed": seed,
                        "code": code,
                    }

            # Verify backend settings were not tampered with during timed eval
            _backend_violations = _check_backend_state()
            if _backend_violations:
                logger.warning(f"Backend settings changed during eval: {_backend_violations}")
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "forbidden pattern detected",
                    "error_code": "execution_failed",
                    "seed": seed,
                    "code": code,
                }

            # Validate miner's returned result
            ok, error, parsed = _validate_return_type(miner_result)
            if not ok or parsed is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": error,
                    "seed": seed,
                    "code": code,
                }

            # Sequence length verification
            expected_seq_len = seq_len - 1  # causal LM uses batch[:, :-1] as input

            if parsed.final_logits is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "final_logits is None - must return actual logits for verification",
                    "error_code": "missing_logits",
                    "seed": seed,
                    "code": code,
                }

            # Verify logits shape is 3D (batch, seq, vocab)
            if len(parsed.final_logits.shape) != 3:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": f"Logits shape invalid: expected 3D (batch, seq, vocab), got shape {parsed.final_logits.shape}",
                    "error_code": "invalid_logits_shape",
                    "seed": seed,
                    "code": code,
                }

            logits_seq_len = parsed.final_logits.shape[1]

            # Verify sequence length matches expected
            if logits_seq_len != expected_seq_len:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": f"Sequence length mismatch (possible truncation): expected {expected_seq_len}, got {logits_seq_len}",
                    "error_code": "sequence_truncation",
                    "seed": seed,
                    "code": code,
                }
            logger.info(f"[PASSED] Sequence length check: {logits_seq_len} == {expected_seq_len}")

            # Verify vocab dimension matches reference (catches fake logits
            # that return e.g. input IDs reshaped to (batch, seq, 1) to avoid
            # materializing the full vocab-sized tensor).
            expected_vocab = reference.final_logits.shape[2]
            actual_vocab = parsed.final_logits.shape[2]
            if actual_vocab != expected_vocab:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": f"Logits vocab dimension mismatch: expected {expected_vocab}, got {actual_vocab}",
                    "error_code": "invalid_logits_shape",
                    "seed": seed,
                    "code": code,
                }
            logger.info(f"[PASSED] Vocab dimension check: {actual_vocab} == {expected_vocab}")

            # Non-rank-0 processes have no reference state and cannot run
            # weight verification.  Return a stub so they reach the finally
            # block (and dist.barrier) quickly instead of blocking rank 0's
            # CPU-bound verification past the NCCL timeout.
            if _multi_gpu and not _IS_RANK_0:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "non-rank-0 stub (rank 0 has real result)",
                    "seed": seed,
                }

            # Build a CPU state dict of the miner's trained model for
            # verification.  final_state is REQUIRED for ALL strategies —
            # miners must return a full state_dict on CPU.
            # Re-use the already-sanitized _mat_state (from line ~2765) instead
            # of re-reading miner_result.final_state, which could trigger
            # deferred computation via __getattr__ on a second access.
            miner_final_state = _mat_state
            if miner_final_state is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "final_state is required for weight verification",
                    "error_code": "missing_final_state",
                    "seed": seed,
                    "code": code,
                }
            if miner_final_state is not None:
                logger.info("Using miner-provided final_state for weight verification")
                expected_keys = set(initial_state.keys())
                provided_keys = set(miner_final_state.keys())
                missing_keys = expected_keys - provided_keys
                extra_keys = provided_keys - expected_keys
                if missing_keys:
                    n_missing = len(missing_keys)
                    sample = sorted(missing_keys)[:3]
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": wall_time,
                        "success": False,
                        "error": (
                            f"final_state missing {n_missing}/{len(expected_keys)} "
                            f"keys (e.g. {sample}). Must return complete state dict."
                        ),
                        "error_code": "incomplete_final_state",
                        "seed": seed,
                        "code": code,
                    }
                if extra_keys:
                    for ek in extra_keys:
                        del miner_final_state[ek]
                    logger.warning(
                        f"Stripped {len(extra_keys)} unexpected keys from "
                        f"miner final_state (e.g. {sorted(extra_keys)[:3]})"
                    )
                for _pname, _pval in miner_final_state.items():
                    if _pname in initial_state and _pval.shape != initial_state[_pname].shape:
                        return {
                            "task_id": task_id,
                            "mfu": 0.0,
                            "tps": 0.0,
                            "total_tokens": 0,
                            "wall_time_seconds": wall_time,
                            "success": False,
                            "error": (
                                f"final_state parameter '{_pname}' has wrong shape "
                                f"({list(_pval.shape)} vs expected "
                                f"{list(initial_state[_pname].shape)})"
                            ),
                            "error_code": "invalid_final_state_shape",
                            "seed": seed,
                            "code": code,
                        }
                candidate_state = miner_final_state

            # Verify sufficient parameters changed during training
            params_ok, params_error, params_details = _verify_params_changed(
                candidate_state, initial_state, min_params_changed_ratio
            )
            if not params_ok:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": params_error,
                    "error_code": "insufficient_params_changed",
                    "seed": seed,
                    "code": code,
                    "diagnostics": {"params_changed": params_details},
                }

            # Calculate expected tokens
            expected_tokens = batch_size * seq_len * steps

            # Verify outputs using weight-based verification
            verified, verify_error, verify_details = _verify_outputs(
                reference,
                parsed,
                expected_tokens,
                reference_final_state=reference_final_state,
                candidate_final_state=candidate_state,
                max_loss_difference=max_loss_difference,
                weight_relative_error_max=weight_relative_error_max,
            )

            tokens_per_rank = expected_tokens
            total_unique_tokens = tokens_per_rank * par_config.dp_size
            tps = float(total_unique_tokens) / max(wall_time, 1e-6)
            mfu = _calculate_mfu(
                total_unique_tokens, wall_time, model_params, gpu_peak_tflops, num_gpus
            )

            # MFU sanity cap — no legitimate code can exceed this on current hardware.
            # Safety net: even if a novel timing attack evades all other checks,
            # physically impossible MFU values are rejected.
            if mfu > max_plausible_mfu:
                logger.warning(f"MFU {mfu:.1f}% exceeds plausible maximum {max_plausible_mfu}%")
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": f"MFU {mfu:.1f}% exceeds plausible maximum {max_plausible_mfu}%",
                    "error_code": "implausible_mfu",
                    "seed": seed,
                    "code": code,
                }

            if verified and mfu < min_mfu:
                logger.warning(f"MFU {mfu:.1f}% below minimum threshold {min_mfu}%")
                return {
                    "task_id": task_id,
                    "mfu": mfu,
                    "tps": tps,
                    "total_tokens": total_unique_tokens,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": f"MFU {mfu:.1f}% below minimum threshold {min_mfu}%",
                    "error_code": "insufficient_mfu",
                    "seed": seed,
                    "code": code,
                }

            # Diagnostics
            diagnostics = {
                "verification": verify_details,
                "strategy": {"dp_size": par_config.dp_size, "tp_size": par_config.tp_size},
                "reference_loss": reference.final_loss,
                "candidate_loss": parsed.final_loss,
                "expected_tokens": expected_tokens,
                "actual_tokens": parsed.total_tokens,
                "total_unique_tokens": total_unique_tokens,
                "model_params": model_params,
                "gpu_peak_tflops": gpu_peak_tflops,
                "trainable_params": trainable_details,
                "params_changed": params_details,
            }

            # Extract error_code from verification details if present
            error_code = None
            if not verified and verify_details:
                # Check gradient_verification sub-details
                grad_details = verify_details.get("gradient_verification", {})
                error_code = grad_details.get("error_code")
                # Check weight_verification sub-details
                if not error_code:
                    weight_details = verify_details.get("weight_verification", {})
                    error_code = weight_details.get("error_code")
                # Check other verification failures
                if not error_code:
                    for check in verify_details.get("checks_failed", []):
                        if check.get("check") == "token_count":
                            error_code = "token_count_mismatch"
                        elif check.get("check") == "loss_comparison":
                            error_code = "loss_mismatch"
                        elif check.get("check") == "weight_verification":
                            error_code = "weight_mismatch"

            return {
                "task_id": task_id,
                "mfu": mfu if verified else 0.0,
                "tps": tps if verified else 0.0,
                "total_tokens": total_unique_tokens if verified else 0,
                "wall_time_seconds": wall_time,
                "success": verified,
                "error": verify_error,
                "error_code": error_code,
                "seed": seed,
                "diagnostics": diagnostics,
                "code": code,
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            if _multi_gpu:
                # In multi-GPU mode, other ranks are blocked in NCCL collectives
                # (allreduce, barrier) waiting for this rank. Graceful cleanup is
                # impossible — the finally block's dist.barrier() would deadlock.
                # Kill immediately so torchrun propagates the failure via SIGTERM.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "seed": seed,
                "code": code,
            }

        finally:
            # Free model gradients to release VRAM held by .grad tensors
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass

            # Explicitly delete large objects to free VRAM/RAM before GC.
            # Without this, local references keep optimizer states (~2x model
            # VRAM), data tensors, gradient copies, and reference state alive,
            # preventing gc.collect() + empty_cache() from reclaiming memory.
            optimizer_miner = None  # type: ignore[assignment]
            data_iter_miner = data = None  # type: ignore[assignment]
            reference_final_state = candidate_state = None  # type: ignore[assignment]
            miner_result = parsed = miner_module = None  # type: ignore[assignment]
            reference = None  # type: ignore[assignment]

            # Always invalidate model cache between evaluations.  FSDP/TP
            # corrupt parameter storage, and DDP/single-GPU miners can leave
            # hooks or modified buffers.  Fresh model each run is safest.
            _CACHE.pop("model", None)

            # Reset torch state (removes miner module, restores timing functions)
            _reset_torch_state()

            # Free torch.compile / CUDA Graph caches from miner's code.
            # This MUST happen here (between evaluations) but NOT between
            # warmup and timed run, so the compiled model persists for timing.
            try:
                torch._dynamo.reset()
            except Exception as e:
                logger.error("torch._dynamo.reset() failed - failing closed", exc_info=True)
                raise RuntimeError("torch._dynamo.reset() failed; runtime state untrusted") from e

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log_vram("after-cleanup")

            # All ranks converge before teardown (60s timeout prevents hangs).
            try:
                if dist.is_initialized():
                    dist.barrier()
                    dist.destroy_process_group()
            except Exception:
                try:
                    if dist.is_initialized():
                        dist.destroy_process_group()
                except Exception:
                    pass


# =============================================================================
# FastAPI HTTP Server (for Basilica custom Docker deployment)
# =============================================================================

app = FastAPI(title="Templar MFU Evaluation", version="2.0.0")

# ---------------------------------------------------------------------------
# In-memory jobs table for async evaluation
# ---------------------------------------------------------------------------
# Maps job_id -> {"status": "pending"|"done"|"failed", "result": dict|None}
_jobs: dict[str, dict] = {}
_jobs_lock = asyncio.Lock()

# Global actor instance (reused for efficiency)
_actor: Actor | None = None


def get_actor() -> Actor:
    """Get or create singleton Actor instance."""
    global _actor
    if _actor is None:
        _actor = Actor()
    return _actor


class EvaluateRequest(BaseModel):
    """Request body for /evaluate endpoint."""

    task_id: int = 0
    seed: str = "0:0:0"
    model_url: str
    data_url: str
    steps: int = 5
    batch_size: int = 8
    timeout: int = 600
    sequence_length: int | None = None
    data_samples: int = 10000
    code: str  # Miner's train.py code
    # Verification settings
    max_loss_difference: float = 0.3
    use_random_init: bool = True
    min_trainable_params_ratio: float = 1.0
    min_params_changed_ratio: float = 0.75
    # Weight verification (0.8% — micro-batch=8 reference keeps FP drift low)
    weight_relative_error_max: float = 0.008
    # Timer integrity
    timer_divergence_threshold: float = 0.005
    # MFU calculation
    gpu_peak_tflops: float = 312.0
    max_plausible_mfu: float = 75.0
    min_mfu: float = 50.0
    # Multi-GPU
    num_gpus: int = Field(default=1, ge=1)


class EvaluateResponse(BaseModel):
    """Response body from /evaluate endpoint."""

    task_id: int
    mfu: float  # Model FLOPs Utilization (primary metric)
    tps: float  # Tokens per second (secondary metric)
    total_tokens: int
    wall_time_seconds: float
    success: bool
    error: str | None = None
    error_code: str | None = None
    seed: str
    diagnostics: dict = {}


@app.get("/health")
async def health():
    """Health check endpoint for Basilica."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


_current_torchrun: asyncio.subprocess.Process | None = None


def _get_descendant_pids(pid: int) -> list[int]:
    """Recursively collect all descendant PIDs via /proc before killing."""
    descendants: list[int] = []
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            child_pids = [int(p) for p in f.read().split()]
        for cpid in child_pids:
            descendants.append(cpid)
            descendants.extend(_get_descendant_pids(cpid))
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, OSError):
        pass
    return descendants


def _kill_torchrun_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL a torchrun process, its process group, AND all descendants.

    torchrun's elastic agent may spawn workers in a different process group
    than the launcher, so os.killpg alone is insufficient.  We walk
    /proc/<pid>/children first (before any kill) to collect every descendant,
    then kill the process group *and* each descendant individually.
    """
    import signal

    if proc.returncode is not None:
        return

    pid = proc.pid

    desc_pids = _get_descendant_pids(pid)

    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
        logger.warning(f"Killed torchrun process group (pgid={pgid})")
    except (ProcessLookupError, PermissionError, OSError):
        pass

    killed_extra = 0
    for dpid in desc_pids:
        try:
            os.kill(dpid, signal.SIGKILL)
            killed_extra += 1
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if killed_extra:
        logger.warning(f"Also killed {killed_extra} descendant processes of torchrun (pid={pid})")

    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


async def _evaluate_via_torchrun(request: EvaluateRequest) -> dict:
    """Spawn torchrun for multi-GPU Basilica evaluation (uvicorn is single-process)."""
    global _current_torchrun
    import asyncio as _aio
    import json as _json
    import tempfile

    if _current_torchrun is not None and _current_torchrun.returncode is None:
        logger.warning(
            f"Killing stale torchrun (pid={_current_torchrun.pid}) before new evaluation"
        )
        _kill_torchrun_group(_current_torchrun)
        try:
            await _aio.wait_for(_current_torchrun.wait(), timeout=10)
        except TimeoutError:
            logger.warning("Stale torchrun launcher did not exit within 10s after kill")
        await _aio.sleep(30)
    _current_torchrun = None

    params_path = None
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir="/tmp", delete=False) as f:
            _json.dump(request.model_dump(), f)
            params_path = f.name

        eval_script = f'''
import asyncio, json, os, sys
sys.path.insert(0, '/app')
from env import Actor

async def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    with open("{params_path}") as f:
        p = json.load(f)
    actor = Actor()
    result = await actor.evaluate(
        task_id=p["task_id"], seed=p["seed"],
        model_url=p["model_url"], data_url=p["data_url"],
        steps=p["steps"], batch_size=p["batch_size"],
        timeout=p["timeout"], sequence_length=p.get("sequence_length"),
        data_samples=p["data_samples"], code=p["code"],
        max_loss_difference=p["max_loss_difference"],
        use_random_init=p["use_random_init"],
        min_trainable_params_ratio=p["min_trainable_params_ratio"],
        min_params_changed_ratio=p["min_params_changed_ratio"],
        weight_relative_error_max=p["weight_relative_error_max"],
        timer_divergence_threshold=p["timer_divergence_threshold"],
        gpu_peak_tflops=p["gpu_peak_tflops"],
        max_plausible_mfu=p["max_plausible_mfu"],
        min_mfu=p["min_mfu"],
        require_cuda_timing=True,
        num_gpus=p["num_gpus"],
    )
    if local_rank == 0:
        print("EVAL_RESULT:" + json.dumps(result))

asyncio.run(main())
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=False) as f:
            f.write(eval_script)
            script_path = f.name

        master_port = 29500 + random.randint(0, 10000)
        proc = await _aio.create_subprocess_exec(
            "torchrun",
            "--nproc_per_node",
            str(request.num_gpus),
            "--master_port",
            str(master_port),
            script_path,
            stdout=_aio.subprocess.PIPE,
            stderr=_aio.subprocess.STDOUT,
            start_new_session=True,
        )
        _current_torchrun = proc

        collected_lines: list[str] = []

        async def _read_and_tee():
            assert proc.stdout is not None
            async for raw_line in proc.stdout:
                line = raw_line.decode(errors="replace").rstrip("\n")
                print(line, flush=True)
                collected_lines.append(line)

        await _aio.wait_for(
            _aio.gather(_read_and_tee(), proc.wait()),
            timeout=request.timeout + 600,
        )
        stdout_text = "\n".join(collected_lines)

        for line in collected_lines:
            if line.startswith("EVAL_RESULT:"):
                return _json.loads(line[len("EVAL_RESULT:") :])

        return {
            "task_id": request.task_id,
            "success": False,
            "error": f"No EVAL_RESULT in torchrun output (exit {proc.returncode}): {stdout_text[-1000:]}",
            "seed": request.seed,
            "mfu": 0.0,
            "tps": 0.0,
            "total_tokens": 0,
            "wall_time_seconds": 0.0,
        }
    except TimeoutError:
        if _current_torchrun is not None:
            _kill_torchrun_group(_current_torchrun)
        return {
            "task_id": request.task_id,
            "success": False,
            "error": f"torchrun evaluation timed out after {request.timeout}s",
            "seed": request.seed,
            "mfu": 0.0,
            "tps": 0.0,
            "total_tokens": 0,
            "wall_time_seconds": 0.0,
        }
    except Exception as e:
        if _current_torchrun is not None:
            _kill_torchrun_group(_current_torchrun)
        return {
            "task_id": request.task_id,
            "success": False,
            "error": f"torchrun evaluation failed: {e}",
            "seed": request.seed,
            "mfu": 0.0,
            "tps": 0.0,
            "total_tokens": 0,
            "wall_time_seconds": 0.0,
        }
    finally:
        for p in (params_path, script_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


async def _run_evaluation(request: EvaluateRequest) -> dict:
    """Run the actual evaluation (sync helper used by background task)."""
    if request.num_gpus > 1:
        return await _evaluate_via_torchrun(request)

    actor = get_actor()
    return await actor.evaluate(
        task_id=request.task_id,
        seed=request.seed,
        model_url=request.model_url,
        data_url=request.data_url,
        steps=request.steps,
        batch_size=request.batch_size,
        timeout=request.timeout,
        sequence_length=request.sequence_length,
        data_samples=request.data_samples,
        code=request.code,
        max_loss_difference=request.max_loss_difference,
        use_random_init=request.use_random_init,
        min_trainable_params_ratio=request.min_trainable_params_ratio,
        min_params_changed_ratio=request.min_params_changed_ratio,
        weight_relative_error_max=request.weight_relative_error_max,
        timer_divergence_threshold=request.timer_divergence_threshold,
        gpu_peak_tflops=request.gpu_peak_tflops,
        max_plausible_mfu=request.max_plausible_mfu,
        min_mfu=request.min_mfu,
        require_cuda_timing=True,
        num_gpus=request.num_gpus,
    )


async def _evaluation_background(job_id: str, request: EvaluateRequest) -> None:
    """Background coroutine: runs evaluation and stores result in _jobs."""
    try:
        result = await _run_evaluation(request)
        async with _jobs_lock:
            _jobs[job_id] = {"status": "done", "result": result}
        logger.info(f"[JOB {job_id}] Evaluation finished successfully")
    except Exception as exc:
        error_result = {
            "task_id": request.task_id,
            "success": False,
            "error": f"Background evaluation failed: {exc}",
            "seed": request.seed,
            "mfu": 0.0,
            "tps": 0.0,
            "total_tokens": 0,
            "wall_time_seconds": 0.0,
        }
        async with _jobs_lock:
            _jobs[job_id] = {"status": "failed", "result": error_result}
        logger.error(f"[JOB {job_id}] Evaluation failed: {exc}")


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """Accept evaluation request, start in background, return job_id immediately.

    Returns HTTP 202 with {"job_id": "..."} so the caller can poll
    GET /eval-status/{job_id} for results. This avoids proxy timeouts
    on long-running evaluations.
    """
    job_id = _uuid.uuid4().hex
    async with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "result": None}
    asyncio.create_task(_evaluation_background(job_id, request))
    logger.info(
        f"[JOB {job_id}] Evaluation accepted (task_id={request.task_id}, "
        f"num_gpus={request.num_gpus})"
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


@app.get("/eval-status/{job_id}")
async def eval_status(job_id: str):
    """Poll for evaluation result.

    Returns:
        - 200 {"status": "pending"}              while evaluation is running
        - 200 {"status": "done", "result": {...}} when evaluation is complete
        - 200 {"status": "failed", "result": {...}} on evaluation error
        - 404 {"error": "unknown job_id"}         if job_id is not found
    """
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "unknown job_id"})

    if job["status"] == "pending":
        return {"status": "pending"}

    result = job["result"]
    response = EvaluateResponse(
        task_id=result.get("task_id", 0),
        mfu=result.get("mfu", 0.0),
        tps=result.get("tps", 0.0),
        total_tokens=result.get("total_tokens", 0),
        wall_time_seconds=result.get("wall_time_seconds", 0.0),
        success=result.get("success", False),
        error=result.get("error"),
        error_code=result.get("error_code"),
        seed=result.get("seed", ""),
        diagnostics=result.get("diagnostics", {}),
    )
    # Clean up to avoid unbounded memory growth
    async with _jobs_lock:
        _jobs.pop(job_id, None)
    return {"status": job["status"], "result": response.model_dump()}


# Entry point when running directly (for local testing)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
