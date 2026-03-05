"""
Shared security policy constants and detection helpers for miner code validation.

Single source of truth consumed by the production validator
(environments/templar/env.py) and the local simulator
(local_test/simulate_validator.py).  Any additions or removals
should be made here so everything stays in sync automatically.
"""

from __future__ import annotations

import ast

# Upper bound on the number of elements in a bytes([...]) / bytearray([...])
# literal the scanner will attempt to decode.  Anything larger is rejected
# outright as a potential resource-exhaustion / DoS vector.
_MAX_BYTES_LITERAL_ELTS = 4096


class SuspiciousConstructionError(Exception):
    """Raised by decode resolvers when the construction looks intentionally
    obfuscated but cannot be statically resolved (e.g. dynamic encoding).

    Callers should treat this as fail-closed: reject the code.
    """


# ---------------------------------------------------------------------------
# Forbidden string patterns
# ---------------------------------------------------------------------------
# Scanned inside string literals, bytes().decode(), str.join(), concatenation,
# f-strings, %-formatting, and attribute names throughout the AST.

FORBIDDEN_STRINGS: list[str] = [
    # Module identity — reassignment enables main-guard code execution
    "__name__",
    "__main__",
    # Dunder escape hatches
    "__setattr__",
    "__delattr__",
    # "__class__" — allowed: miners need obj.__class__ for FSDP layer detection
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__import__",
    "__builtins__",
    "__dict__",
    "__globals__",
    "__code__",
    "__func__",
    "__self__",
    "__init_subclass__",
    "__traceback__",
    "__getattribute__",
    "__build_class__",
    # Timer / profiling internals
    "perf_counter",
    "_perf_counter",
    "monotonic",
    "_monotonic",
    "_REAL_PC_ID",
    "_REAL_MONO_ID",
    "_REAL_SYNC_ID",
    "_cuda_synchronize",
    # CUDA event timing
    "elapsed_time",
    "_et_orig",
    "_cuda_elapsed_time",
    "_REAL_ET_ID",
    # Cross-entropy reference poisoning
    "_real_cross_entropy",
    "_REAL_CE_ID",
    # GC / introspection
    "get_objects",
    "get_referrers",
    "get_referents",
    # Validator optimizer internals
    "captured_gradients",
    "_opt_impl",
    "_grad_snapshot_gpu",
    "step_count",
    "GradientCapturingOptimizer",
    # Dynamic import / module access
    "importlib",
    "import_module",
    "sys.modules",
    # Dangerous builtins (as strings for obfuscation detection)
    "setattr",
    "getattr",
    "delattr",
    "globals",
    "locals",
    "builtins",
    # Frame / traceback traversal
    "tb_frame",
    "tb_next",
    "f_globals",
    "f_builtins",
    "f_locals",
    "f_back",
    "co_consts",
    "co_names",
    # Generator / coroutine / async-generator frame access
    "gi_frame",
    "gi_code",
    "cr_frame",
    "cr_code",
    "ag_frame",
    "ag_code",
    # Operator helpers that can bypass attribute guards
    "operator.attrgetter",
    "operator.methodcaller",
    "attrgetter",
    "methodcaller",
    # Torch internal access (even under aliases)
    "_C",
    "_dynamo",
    "_inductor",
    # Validator env internals
    "_CACHE",
    "initial_state",
    "_hidden_modules",
    "_sensitive_keys",
    # unittest.mock attribute patching
    "mock.patch",
    "MagicMock",
    "unittest.mock",
]

# ---------------------------------------------------------------------------
# Forbidden module imports
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES: set[str] = {
    "ctypes",
    "_ctypes",
    "gc",
    "subprocess",
    "sys",
    "os",
    "pathlib",
    "io",
    "_io",
    "socket",
    "http",
    "urllib",
    "requests",
    "shutil",
    "tempfile",
    "signal",
    "threading",
    "_thread",
    "multiprocessing",
    "inspect",
    "ast",
    "dis",
    "code",
    "codeop",
    "compileall",
    "pickle",
    "shelve",
    "marshal",
    "builtins",
    "_builtins",
    "operator",
    "types",
    "codecs",
    "base64",
    "pdb",
    "pprint",
    "runpy",
    "linecache",
    "pkgutil",
    "atexit",
    "site",
    "zipimport",
    # Stack inspection — reveals internal file paths and code structure
    "traceback",
    # Block importing the validator's own module (timer tampering attack)
    "env",
    "time",
    "__main__",
    "miner_train",
    # logging.Logger.manager.loggerDict holds refs to all loggers —
    # traversable to reach the env module's logger and back to env itself
    "logging",
    # unittest.mock.patch can replace any attribute on any object
    "unittest",
    "mock",
    # functools.partial can wrap forbidden functions — ALLOWED: miners need
    # functools.partial for FSDP auto_wrap_policy and similar optimisations.
    # "functools",
    # weakref can observe GC behavior and hold references
    "weakref",
}

# Dotted module paths blocked even when the base module is allowed
FORBIDDEN_DOTTED_MODULES: set[str] = {
    "torch.utils.dlpack",
    "numpy.ctypeslib",
}

# Substring checked against import paths to block C++ extension compilation
FORBIDDEN_IMPORT_SUBSTRINGS: set[str] = {
    "cpp_extension",
}

# ---------------------------------------------------------------------------
# Torch-specific import / alias guards
# ---------------------------------------------------------------------------

# torch symbols that enable bypassing attribute-based guards when imported directly
FORBIDDEN_TORCH_SYMBOL_IMPORTS: set[str] = {
    "load",
    "compile",
    "_C",
    "_dynamo",
    "_inductor",
}

# Backend toggles that bypass attribute-call checks if imported as bare names
FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS: set[str] = set()

# torch attributes that should never be rebound or called via alias variables
FORBIDDEN_TORCH_ATTRIBUTE_ALIASES: set[str] = {
    "load",
    "compile",
    "_C",
    "_dynamo",
    "_inductor",
}

# torch internal config namespaces whose .config attrs must not be written
FORBIDDEN_TORCH_CONFIG_MODULES: set[str] = {
    "_dynamo",
    "_inductor",
}

# ---------------------------------------------------------------------------
# Forbidden bare-name references
# ---------------------------------------------------------------------------
# Builtins like __import__, exec, setattr are blocked as CALLS, but a miner
# can bypass by storing a reference: `_imp = __import__; _imp("sys")`.
# This set blocks them even when only *referenced* (ast.Name in Load context).

FORBIDDEN_NAMES: set[str] = {
    "__import__",
    "exec",
    "eval",
    "compile",
    "breakpoint",
    "setattr",
    "getattr",
    "delattr",
    "vars",
    "dir",
    "globals",
    "locals",
    # "type" — allowed: miners need type() for FSDP auto_wrap_policy layer detection
    "memoryview",
    "open",
    "chr",
    "ord",
    "input",
    "classmethod",
    "staticmethod",
    "property",
    "__build_class__",
}

# Subset of builtins blocked when called (overlaps with FORBIDDEN_NAMES
# but kept separate for the call-site check).
FORBIDDEN_BUILTINS: set[str] = {
    "setattr",
    "getattr",
    "delattr",
    "vars",
    "dir",
    "globals",
    "locals",
    # "type" — allowed: miners need type() for FSDP auto_wrap_policy layer detection
    "memoryview",
    "open",
    "chr",
    "ord",
    "breakpoint",
    "input",
    "classmethod",
    "staticmethod",
    "property",
}

# Dangerous functions blocked as direct calls: name(...) form
FORBIDDEN_DIRECT_CALLS: set[str] = {
    "exec",
    "eval",
    "compile",
    "__import__",
}

# Dangerous functions blocked as attribute calls: obj.attr(...) form
FORBIDDEN_ATTR_CALLS: set[str] = {
    "exec",
    "eval",
    "__import__",
}

# ---------------------------------------------------------------------------
# Timer / timing-related attribute guards
# ---------------------------------------------------------------------------

# Attribute names blocked on ANY object (ast.Attribute check)
FORBIDDEN_TIMER_ATTRS: set[str] = {
    "perf_counter",
    "_perf_counter",
    "monotonic",
    "_monotonic",
    "perf_counter_ns",
    "monotonic_ns",
    "_cuda_synchronize",
    "elapsed_time",
    "_et_orig",
}

# Attribute names that must never appear as assignment targets
FORBIDDEN_ASSIGNMENT_ATTRS: set[str] = {
    "synchronize",
    "elapsed_time",
}

# Allowed torch submodule import paths.  Only these may be imported via
# ``import torch.X.Y as Z`` or ``from torch.X import Y``.  The local
# alias (Z / Y) is tracked so that *attribute assignment* on it
# (e.g. ``F.cross_entropy = fake``) is blocked — legitimate optimisation
# never needs to monkey-patch torch internals.
ALLOWED_TORCH_SUBMODULE_IMPORTS: set[str] = {
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.parallel",
    "torch.amp",
    "torch.cuda",
    "torch.cuda.amp",
    "torch.distributed",
    "torch.distributed.fsdp",
    "torch.distributed.fsdp.wrap",
    "torch.distributed.tensor",
    "torch.distributed.tensor.parallel",
    "torch.distributed.device_mesh",
    "torch.distributed.optim",
    "torch.utils.checkpoint",
}

# Prefixes of ``torch.*`` dotted paths where attribute *assignment* is
# permitted.  Everything else under ``torch.*`` is blocked to prevent
# monkey-patching (e.g. ``torch.autograd.backward = fake``).
# Currently only ``torch.backends.*`` is needed for legitimate config
# such as ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction``.
ALLOWED_TORCH_ASSIGNMENT_PREFIXES: set[str] = {
    "torch.backends",
}

# Gradient-control functions that must never be called — disabling
# gradients would make backward() a no-op and bypass gradient verification.
FORBIDDEN_GRAD_TOGGLE_CALLS: set[str] = {
    "set_grad_enabled",
    "inference_mode",
}

# ---------------------------------------------------------------------------
# Attribute-level AST guards
# ---------------------------------------------------------------------------

# object.__setattr__ / object.__delattr__ — blocked on `object` receiver
FORBIDDEN_OBJECT_DUNDER_ATTRS: set[str] = {
    "__setattr__",
    "__delattr__",
}

# gc.get_objects() etc. — blocked when receiver is `gc`
FORBIDDEN_GC_ATTRS: set[str] = {
    "get_objects",
    "get_referrers",
    "get_referents",
}

# cuDNN backend settings — blocked as Store targets
FORBIDDEN_CUDNN_ATTRS: set[str] = {
    "deterministic",
    "benchmark",
}

# Backend toggle / precision calls — no longer blocked; verification
# catches any numerical divergence and miners bear the consequences of
# changing precision settings.  Kept as empty set for backward compat.
FORBIDDEN_BACKEND_TOGGLE_ATTRS: set[str] = set()

# sys.modules access — blocked when receiver matches these names
FORBIDDEN_SYS_MODULE_NAMES: set[str] = {
    "sys",
    "_sys",
}

# Dunder / frame / traceback attributes blocked on any object
FORBIDDEN_INTROSPECTION_ATTRS: set[str] = {
    "__globals__",
    "__code__",
    "__func__",
    "__self__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__init_subclass__",
    "__traceback__",
    "tb_frame",
    "tb_next",
    "f_globals",
    "f_builtins",
    "f_locals",
    "f_code",
    "f_back",
    "co_consts",
    "co_names",
    "__getattribute__",
    # Generator / coroutine / async-generator frame access
    "gi_frame",
    "gi_code",
    "cr_frame",
    "cr_code",
    "ag_frame",
    "ag_code",
}


# ---------------------------------------------------------------------------
# Shared AST detection helpers
# ---------------------------------------------------------------------------
# Used by env.py so detection logic is maintained in one place.
# Scanners import these and call them during AST walks.


def forbidden_name_binding_reason(node: ast.AST) -> str | None:
    """Return violation text when ``node`` binds or modifies ``__name__``."""
    target = "__name__"

    if (
        isinstance(node, ast.Name)
        and node.id == target
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ):
        return "modification of __name__ is forbidden"

    if isinstance(node, ast.alias):
        if node.asname == target:
            return "aliasing import to __name__ is forbidden"
        if node.name == target:
            return "importing __name__ is forbidden"

    if isinstance(node, ast.arg) and node.arg == target:
        return "using __name__ as an argument is forbidden"

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if node.name == target:
            return "defining __name__ is forbidden"

    if isinstance(node, ast.ExceptHandler) and node.name == target:
        return "binding __name__ in exception handler is forbidden"

    if isinstance(node, ast.MatchAs) and getattr(node, "name", None) == target:
        return "binding __name__ in match pattern is forbidden"

    if isinstance(node, ast.MatchStar) and getattr(node, "name", None) == target:
        return "binding __name__ in match pattern is forbidden"

    if isinstance(node, ast.MatchMapping) and getattr(node, "rest", None) == target:
        return "binding __name__ in match mapping pattern is forbidden"

    if isinstance(node, ast.Global) and target in node.names:
        return "declaring global __name__ is forbidden"

    if isinstance(node, ast.Nonlocal) and target in node.names:
        return "declaring nonlocal __name__ is forbidden"

    if hasattr(ast, "TypeVar") and isinstance(node, ast.TypeVar):
        if node.name == target:
            return "using __name__ as a type parameter is forbidden"

    return None


def _literal_str_arg(node: ast.Call, pos: int = 0, name: str | None = None) -> str | None:
    """Extract a literal string argument from an AST Call node.

    Checks positional arg at *pos* first, then keyword arg named *name*.
    Returns the string value or ``None``.
    """
    if node.args and len(node.args) > pos:
        arg = node.args[pos]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    if name:
        for kw in node.keywords:
            if (
                kw.arg == name
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                return kw.value.value
    return None


def _decode_with_encoding(raw: bytes, enc_node: ast.Call) -> str | None:
    """Decode *raw* bytes using encoding extracted from *enc_node* (.decode() call).

    Honors both positional and keyword ``encoding`` / ``errors`` arguments.
    Falls back to UTF-8 when no encoding is specified.  Raises
    :class:`SuspiciousConstructionError` if an encoding/errors argument is present
    but not a constant string (fail-closed).
    """
    has_encoding_arg = bool(enc_node.args) or any(kw.arg == "encoding" for kw in enc_node.keywords)
    encoding = _literal_str_arg(enc_node, pos=0, name="encoding")
    if has_encoding_arg and encoding is None:
        raise SuspiciousConstructionError("dynamic .decode(encoding) construction")
    encoding = encoding or "utf-8"

    has_errors_arg = len(enc_node.args) >= 2 or any(kw.arg == "errors" for kw in enc_node.keywords)
    errors = _literal_str_arg(enc_node, pos=1, name="errors")
    if has_errors_arg and errors is None:
        raise SuspiciousConstructionError("dynamic .decode(..., errors) construction")
    errors = errors or "strict"

    try:
        return raw.decode(encoding, errors)
    except (ValueError, UnicodeDecodeError, LookupError):
        return None


def try_decode_bytes_node(node: ast.AST) -> str | None:
    """Try to statically resolve a ``.decode()`` call on bytes to a string.

    Handles: ``bytes([...]).decode()``, ``bytearray([...]).decode()``,
    ``bytes.fromhex("...").decode()``, ``(b'x' + b'y').decode()``,
    and ``b'literal'.decode()``.  Honors explicit ``encoding`` and
    ``errors`` parameters.  Returns the decoded string or None.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "decode"
    ):
        return None

    inner = node.func.value

    # bytes([int, ...]).decode() / bytearray([int, ...]).decode()
    if (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id in ("bytes", "bytearray")
        and inner.args
        and len(inner.args) == 1
        and isinstance(inner.args[0], ast.List)
    ):
        if len(inner.args[0].elts) > _MAX_BYTES_LITERAL_ELTS:
            raise SuspiciousConstructionError("bytes literal too large")
        int_values = []
        for elt in inner.args[0].elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                int_values.append(elt.value)
            else:
                return None
        try:
            raw = bytes(int_values)
        except ValueError:
            return None
        return _decode_with_encoding(raw, node)

    # bytes.fromhex("hex").decode() / bytearray.fromhex("hex").decode()
    if (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "fromhex"
        and isinstance(inner.func.value, ast.Name)
        and inner.func.value.id in ("bytes", "bytearray")
        and inner.args
        and len(inner.args) == 1
        and isinstance(inner.args[0], ast.Constant)
        and isinstance(inner.args[0].value, str)
    ):
        try:
            raw = bytes.fromhex(inner.args[0].value)
        except ValueError:
            return None
        return _decode_with_encoding(raw, node)

    # (b'x' + b'y').decode()
    if (
        isinstance(inner, ast.BinOp)
        and isinstance(inner.op, ast.Add)
        and isinstance(inner.left, ast.Constant)
        and isinstance(inner.left.value, bytes)
        and isinstance(inner.right, ast.Constant)
        and isinstance(inner.right.value, bytes)
    ):
        raw = inner.left.value + inner.right.value
        return _decode_with_encoding(raw, node)

    # b'literal'.decode()
    if isinstance(inner, ast.Constant) and isinstance(inner.value, bytes):
        return _decode_with_encoding(inner.value, node)

    return None


def try_decode_str_bytes_constructor(node: ast.AST) -> str | None:
    """Try to resolve ``str(bytes([...]), 'enc')`` / ``str(bytearray([...]), 'enc')``.

    Honors explicit encoding and errors parameters (positional and keyword).
    Returns the decoded string or None.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) >= 1
    ):
        return None

    inner = node.args[0]
    if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)):
        return None
    if inner.func.id not in ("bytes", "bytearray"):
        return None

    # Fail-closed: if there's an encoding arg but it's not a constant string,
    # the miner is hiding the encoding dynamically — reject outright.
    has_encoding_arg = len(node.args) >= 2 or any(kw.arg == "encoding" for kw in node.keywords)
    encoding = _literal_str_arg(node, pos=1, name="encoding")
    if has_encoding_arg and encoding is None:
        raise SuspiciousConstructionError("dynamic str(bytes/bytearray, encoding) construction")
    encoding = encoding or "utf-8"

    has_errors_arg = len(node.args) >= 3 or any(kw.arg == "errors" for kw in node.keywords)
    errors = _literal_str_arg(node, pos=2, name="errors")
    if has_errors_arg and errors is None:
        raise SuspiciousConstructionError("dynamic str(bytes/bytearray, ..., errors) construction")
    errors = errors or "strict"

    if not (inner.args and len(inner.args) == 1 and isinstance(inner.args[0], ast.List)):
        return None

    if len(inner.args[0].elts) > _MAX_BYTES_LITERAL_ELTS:
        raise SuspiciousConstructionError("bytes literal too large")

    int_values = []
    for elt in inner.args[0].elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
            int_values.append(elt.value)
        else:
            return None
    try:
        return bytes(int_values).decode(encoding, errors)
    except (ValueError, UnicodeDecodeError, LookupError):
        return None


def try_resolve_join(node: ast.AST) -> str | None:
    """Try to resolve ``"sep".join([...])`` or ``"".join(reversed("..."))``."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
        and node.args
        and len(node.args) == 1
    ):
        return None

    sep = node.func.value.value
    arg = node.args[0]

    if isinstance(arg, (ast.List, ast.Tuple)):
        chars = []
        for elt in arg.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                chars.append(elt.value)
            else:
                return None
        return sep.join(chars) if chars else None

    if (
        isinstance(arg, ast.Call)
        and isinstance(arg.func, ast.Name)
        and arg.func.id == "reversed"
        and arg.args
        and len(arg.args) == 1
        and isinstance(arg.args[0], ast.Constant)
        and isinstance(arg.args[0].value, str)
    ):
        return sep.join(reversed(arg.args[0].value))

    return None


def try_resolve_concat(node: ast.AST) -> str | None:
    """Recursively resolve chained string ``Add`` BinOps: ``"a" + "b" + "c"``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = try_resolve_concat(node.left)
        right = try_resolve_concat(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def try_resolve_format(node: ast.AST) -> str | None:
    """Resolve ``"%s..." % "x"`` and ``"%s%s" % ("x", "y")``."""
    if not (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mod)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, str)
    ):
        return None

    parts: list[str] | None = None
    if isinstance(node.right, ast.Tuple):
        parts = []
        for elt in node.right.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                parts.append(elt.value)
            else:
                return None
    elif isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
        parts = [node.right.value]

    if parts is None:
        return None
    try:
        return node.left.value % tuple(parts)
    except (TypeError, ValueError):
        return None


def try_resolve_fstring(node: ast.AST) -> str | None:
    """Resolve f-strings where all parts are constants: ``f"{'x'}_back"``."""
    if not isinstance(node, ast.JoinedStr):
        return None

    parts = []
    for val in node.values:
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            parts.append(val.value)
        elif (
            isinstance(val, ast.FormattedValue)
            and isinstance(val.value, ast.Constant)
            and isinstance(val.value.value, str)
            and val.format_spec is None
            and val.conversion == -1
        ):
            parts.append(val.value.value)
        else:
            return None
    return "".join(parts) if parts else None
