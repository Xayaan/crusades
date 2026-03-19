"""Comprehensive tests for the Arbos agent components.

Exercises all code paths: env loading, hparams, validation, state persistence,
response parsing, LLM client init, file outputs, and the full agent dry-run loop.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ARBOS_DIR = Path(__file__).parent
sys.path.insert(0, str(ARBOS_DIR))

from agent import (  # noqa: E402
    AgentState,
    Attempt,
    format_diagnostics,
    format_history,
    load_hparams,
    save_best,
    validate_code,
)
from llm_client import (  # noqa: E402
    LLMResponse,
    _load_system_prompt,
    _parse_response,
    _strip_thinking_tags,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  {PASS}  {name}")
    else:
        failed += 1
        msg = f" — {detail}" if detail else ""
        print(f"  {FAIL}  {name}{msg}")


def test_hparams():
    print("\n--- hparams loading ---")
    h = load_hparams()
    check("hparams loads", isinstance(h, dict))
    check("has benchmark_model_name", "benchmark_model_name" in h)
    check("has verification", "verification" in h)
    check("has mfu", "mfu" in h)
    check("has basilica", "basilica" in h)
    check("eval_steps is int", isinstance(h.get("eval_steps"), int))
    check("gpu_peak_tflops", h["mfu"]["gpu_peak_tflops"] == 312.0)
    check("gpu_count in basilica", h["basilica"]["gpu_count"] == 2)


def test_validate_code():
    print("\n--- code validation ---")
    good_code = """
from dataclasses import dataclass
import torch

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None

def get_strategy():
    return {"dp_size": 2, "tp_size": 1}

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    return InnerStepsResult(
        final_logits=torch.zeros(1),
        total_tokens=100,
        final_loss=0.5,
        final_state=None,
    )
"""
    valid, msg = validate_code(good_code)
    check("valid code passes", valid, msg)

    bad1 = "def get_strategy(): return 'fsdp'"
    valid, msg = validate_code(bad1)
    check("missing inner_steps rejected", not valid)

    bad2 = "def inner_steps(): pass\ndef get_strategy(): pass"
    valid, msg = validate_code(bad2)
    check("missing InnerStepsResult rejected", not valid)

    bad3 = "def inner_steps( InnerStepsResult\ndef get_strategy("
    valid, msg = validate_code(bad3)
    check("syntax error rejected", not valid)
    check("syntax error has message", "Syntax error" in msg)

    # Security scanner catches forbidden patterns
    bad_cudnn = """from dataclasses import dataclass
import torch

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None

def get_strategy():
    return {"dp_size": 2, "tp_size": 1}

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    torch.backends.cudnn.benchmark = True
    return InnerStepsResult(final_logits=torch.zeros(1), total_tokens=100, final_loss=0.5)
"""
    valid, msg = validate_code(bad_cudnn)
    check("cudnn.benchmark rejected by security scanner", not valid)
    check("security violation in message", "Security violation" in msg, msg)


def test_state_persistence():
    print("\n--- state persistence ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "state.json"
        import agent as agent_mod

        orig = agent_mod.STATE_FILE
        agent_mod.STATE_FILE = state_file

        try:
            s = AgentState(best_mfu=55.5, best_code_path="/tmp/test.py", step=3)
            s.add_attempt(Attempt(step=1, reasoning="test opt", mfu=50.0, success=True))
            s.add_attempt(
                Attempt(step=2, reasoning="failed opt", mfu=0.0, success=False, error="bad code")
            )
            s.save()
            check("state file created", state_file.exists())

            loaded = AgentState.load()
            check("state best_mfu preserved", loaded.best_mfu == 55.5)
            check("state step preserved", loaded.step == 3)
            check("state history count", len(loaded.history) == 2)
            check("history attempt data", loaded.history[0]["reasoning"] == "test opt")

            s2 = AgentState()
            for i in range(60):
                s2.add_attempt(
                    Attempt(step=i, reasoning=f"attempt {i}", mfu=float(i), success=True)
                )
            check("history truncated to 50", len(s2.history) == 50)
        finally:
            agent_mod.STATE_FILE = orig


def test_format_diagnostics():
    print("\n--- format_diagnostics ---")
    check("empty diagnostics", format_diagnostics({}) == "")
    check("None diagnostics", format_diagnostics(None) == "")

    diag_with_checks = {
        "verification": {
            "checks_failed": [
                {"check": "loss_mismatch", "error": "Loss difference 0.45 > 0.3"},
            ],
            "gradient_verification": {"relative_error": 0.05},
        },
        "reference_loss": 2.45,
        "candidate_loss": 2.90,
        "strategy": "fsdp",
    }
    text = format_diagnostics(diag_with_checks)
    check("contains check_failed", "loss_mismatch" in text)
    check("contains gradient error", "0.05" in text)
    check("contains reference_loss", "2.45" in text)
    check("contains candidate_loss", "2.9" in text)
    check("contains strategy", "fsdp" in text)

    diag_with_params = {
        "params_changed": {"ratio": 0.82},
        "trainable_params": {"ratio": 1.0},
    }
    text2 = format_diagnostics(diag_with_params)
    check("contains params_changed_ratio", "0.82" in text2)
    check("contains trainable_params_ratio", "1.0" in text2)


def test_format_history():
    print("\n--- format_history ---")
    check("empty history", format_history([]) == "No previous attempts.")

    history = [
        {"step": 1, "reasoning": "optimize lr", "mfu": 50.0, "success": True, "error": None},
        {
            "step": 2,
            "reasoning": "failed attempt",
            "mfu": 0,
            "success": False,
            "error": "syntax error in code",
            "error_code": "execution_failed",
            "diagnostics_text": "  reference_loss: 2.45\n  candidate_loss: 2.90",
            "code_diff": "-old_line\n+new_line",
        },
    ]
    output = format_history(history, best_mfu=50.0)
    check("contains IMPROVED step", "[IMPROVED]" in output)
    check("contains FAIL step", "[FAIL]" in output)
    check("contains MFU value", "50.00%" in output)
    check("contains code diff", "-old_line" in output)
    check("contains error", "syntax error" in output)
    check("contains error_code", "execution_failed" in output)
    check("contains diagnostics", "reference_loss" in output)


def test_save_best():
    print("\n--- save_best ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        import agent as agent_mod

        orig = agent_mod.BEST_DIR
        agent_mod.BEST_DIR = Path(tmpdir)

        try:
            code = "# test code\ndef inner_steps(): pass"
            path = save_best(code, 5, 62.3)
            check("best file created", path.exists())
            check("best file has code", path.read_text() == code)
            check("filename contains step and mfu", "v5_62.3" in path.name)

            latest = Path(tmpdir) / "train_latest.py"
            check("latest symlink created", latest.exists())
            check("latest has same code", latest.read_text() == code)
        finally:
            agent_mod.BEST_DIR = orig


def test_strip_thinking_tags():
    print("\n--- thinking tag stripping ---")
    text_with_think = "<think>I should optimize the learning rate...</think>\nREASONING:\nOptimized LR\n\nCODE:\n```python\ndef inner_steps(): pass\n```"
    stripped = _strip_thinking_tags(text_with_think)
    check("think tag removed", "<think>" not in stripped)
    check("content preserved", "REASONING:" in stripped)

    text_multiline = "<think>\nLine 1\nLine 2\nLine 3\n</think>\nActual output here"
    stripped2 = _strip_thinking_tags(text_multiline)
    check("multiline think removed", "Line 1" not in stripped2)
    check("output preserved", "Actual output here" in stripped2)

    text_no_think = "REASONING:\nTest\n\nCODE:\n```python\nx = 1\n```"
    stripped3 = _strip_thinking_tags(text_no_think)
    check("no think tags unchanged", stripped3 == text_no_think)


def test_parse_response():
    print("\n--- response parsing ---")

    standard = """REASONING:
Switched to SHARD_GRAD_OP for better MFU

CODE:
```python
from dataclasses import dataclass
import torch

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None

def get_strategy():
    return {"dp_size": 2, "tp_size": 1}

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    return InnerStepsResult(
        final_logits=torch.zeros(1),
        total_tokens=100,
        final_loss=0.5,
        final_state=None,
    )
```"""
    result = _parse_response(standard)
    check("standard: reasoning extracted", "SHARD_GRAD_OP" in result.reasoning)
    check("standard: code extracted", "def inner_steps" in result.code)
    check("standard: code has get_strategy", "def get_strategy" in result.code)

    with_think = "<think>Let me think about this carefully. I should try using a different sharding strategy.</think>\nREASONING:\nUsing fused optimizer\n\nCODE:\n```python\ndef get_strategy(): return 'fsdp'\ndef inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):\n    from dataclasses import dataclass\n    import torch\n    @dataclass\n    class InnerStepsResult:\n        final_logits: torch.Tensor\n        total_tokens: int\n        final_loss: float\n        final_state: dict | None = None\n    return InnerStepsResult(torch.zeros(1), 0, 0.0, None)\n```"
    result2 = _parse_response(with_think)
    check("think-tag: reasoning extracted", "fused optimizer" in result2.reasoning)
    check("think-tag: code extracted", "def inner_steps" in result2.code)

    no_code = "REASONING:\nSomething\n\nHere is my suggestion but no code block"
    try:
        _parse_response(no_code)
        check("no-code: raises ValueError", False, "should have raised")
    except ValueError:
        check("no-code: raises ValueError", True)

    fallback = """REASONING:
Testing fallback parser

import torch

def get_strategy():
    return "fsdp"

class InnerStepsResult:
    pass

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    return InnerStepsResult()"""
    result3 = _parse_response(fallback)
    check("fallback: extracts code without fences", "def inner_steps" in result3.code)


def test_llm_client_init():
    print("\n--- LLM client initialization ---")
    from llm_client import LLMClient

    os.environ["CHUTES_API_KEY"] = "test-key-for-init"
    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("LLM_MODEL", None)

    client = LLMClient()
    check("default provider is chutes", client.provider == "chutes")
    check("default model is DeepSeek-R1", client.model == "deepseek-ai/DeepSeek-R1-0528-TEE")
    check("base_url set", "chutes.ai" in client.base_url)
    check("system prompt loaded", len(_load_system_prompt()) > 100)

    os.environ["LLM_PROVIDER"] = "openrouter"
    os.environ["OPENROUTER_API_KEY"] = "test-or-key"
    client2 = LLMClient()
    check("openrouter provider", client2.provider == "openrouter")
    check("openrouter default model", "claude" in client2.model)
    check("openrouter base_url", "openrouter.ai" in client2.base_url)

    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["ANTHROPIC_API_KEY"] = "test-ant-key"
    client3 = LLMClient()
    check("anthropic provider", client3.provider == "anthropic")
    check("anthropic base_url", "anthropic.com" in client3.base_url)

    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("CHUTES_API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)

    try:
        LLMClient()
        check("missing key raises", False, "should have raised")
    except ValueError as e:
        check("missing key raises ValueError", True)
        check("error mentions key name", "CHUTES_API_KEY" in str(e))


def test_tester_payload():
    print("\n--- Basilica tester payload ---")
    from tester import BasilicaTester

    h = load_hparams()
    tester = BasilicaTester(h)
    payload = tester._build_payload("# test code")

    check("payload has model_url", payload["model_url"] == "Qwen/Qwen2.5-3B")
    check("payload has data_url", payload["data_url"] == "HuggingFaceFW/fineweb")
    check("payload has batch_size", payload["batch_size"] == 16)
    check("payload has sequence_length", payload["sequence_length"] == 1024)
    check("payload has steps", payload["steps"] == 20)
    check("payload has code", payload["code"] == "# test code")
    check("payload has num_gpus", payload["num_gpus"] == 2)
    check("payload has gpu_peak_tflops", payload["gpu_peak_tflops"] == 312.0)
    check("payload has max_loss_difference", payload["max_loss_difference"] == 0.3)
    check("payload has min_params_changed_ratio", payload["min_params_changed_ratio"] == 0.75)
    check("payload has weight_relative_error_max", payload["weight_relative_error_max"] == 0.008)
    check("payload has timer_divergence_threshold", payload["timer_divergence_threshold"] == 0.005)


def test_local_tester_payload():
    print("\n--- LocalDocker tester payload ---")
    from tester import LocalDockerTester

    h = load_hparams()
    tester = LocalDockerTester(h, num_gpus=4, gpu_devices="4,5,6,7")
    payload = tester._build_payload("# test code")

    check("local payload model_url", payload["model_url"] == "Qwen/Qwen2.5-3B")
    check("local payload num_gpus=4", payload["num_gpus"] == 4)
    check("local payload has verification fields", payload["weight_relative_error_max"] == 0.008)

    cmd = tester._docker_cmd("/tmp/code.py", "/tmp/params.json", "/tmp/runner.py")
    check("docker cmd starts with docker", cmd[0] == "docker")
    check("docker cmd has --gpus", "--gpus" in cmd)
    check("docker cmd uses device flag", "device=4,5,6,7" in " ".join(cmd))
    check("docker cmd has torchrun for multi-gpu", "torchrun" in cmd)
    check("docker cmd has nproc_per_node=4", "4" in cmd[cmd.index("--nproc_per_node") + 1])

    tester_single = LocalDockerTester(h, num_gpus=1)
    cmd_single = tester_single._docker_cmd("/tmp/c.py", "/tmp/p.json", "/tmp/r.py")
    check("single-gpu uses python3", "python3" in cmd_single)
    check("single-gpu no torchrun", "torchrun" not in cmd_single)


def test_env_loading():
    print("\n--- .env file loading ---")
    env_path = ARBOS_DIR / ".env"
    check(".env file exists", env_path.exists())
    content = env_path.read_text()
    check(".env has BASILICA_API_TOKEN", "BASILICA_API_TOKEN" in content)
    check(".env has CHUTES_API_KEY", "CHUTES_API_KEY" in content)

    example_path = ARBOS_DIR / ".env.example"
    check(".env.example exists", example_path.exists())
    example = example_path.read_text()
    check("example has LLM_PROVIDER", "LLM_PROVIDER" in example)
    check("example has BASILICA_API_TOKEN", "BASILICA_API_TOKEN" in example)


def test_prompt_file():
    print("\n--- PROMPT.md ---")
    prompt_path = ARBOS_DIR / "PROMPT.md"
    check("PROMPT.md exists", prompt_path.exists())
    content = prompt_path.read_text()
    check("prompt has contract", "train.py Contract" in content)
    check("prompt has MFU", "MFU" in content)
    check("prompt has security", "Security" in content)
    check("prompt has verification", "Verification" in content)
    check("prompt has response format", "Response Format" in content)
    check("prompt has forbidden imports", "subprocess" in content)
    check("prompt has allowed imports", "flash_attn" in content)


def test_full_dryrun_with_mock():
    print("\n--- full dry-run with mock LLM ---")
    mock_code = """from dataclasses import dataclass
import torch

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None

def get_strategy():
    return {"dp_size": 2, "tp_size": 1}

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    return InnerStepsResult(
        final_logits=torch.zeros(1),
        total_tokens=100,
        final_loss=0.5,
        final_state=None,
    )
"""
    mock_response = LLMResponse(reasoning="Optimized learning rate", code=mock_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        import agent as agent_mod

        orig_runs = agent_mod.RUNS_DIR
        orig_best = agent_mod.BEST_DIR
        orig_state = agent_mod.STATE_FILE
        orig_log = agent_mod.LOG_DIR

        agent_mod.RUNS_DIR = Path(tmpdir) / "runs"
        agent_mod.BEST_DIR = Path(tmpdir) / "best"
        agent_mod.STATE_FILE = Path(tmpdir) / "state.json"
        agent_mod.LOG_DIR = Path(tmpdir) / "logs"

        try:
            os.environ["CHUTES_API_KEY"] = "test-key"
            os.environ.pop("LLM_PROVIDER", None)
            os.environ.pop("LLM_MODEL", None)

            with patch("llm_client.LLMClient") as mock_llm_cls:
                mock_instance = MagicMock()
                mock_instance.provider = "chutes"
                mock_instance.model = "deepseek-ai/DeepSeek-V3"
                mock_instance.suggest_improvement.return_value = mock_response
                mock_llm_cls.return_value = mock_instance

                class FakeArgs:
                    train_py = str(ARBOS_DIR.parent / "local_test" / "train_fsdp.py")
                    mode = "miner"
                    max_steps = 2
                    dry_run = True
                    local = False
                    num_gpus = None
                    gpu_devices = None
                    docker_image = None
                    env_file = None

                agent_mod.run_agent(FakeArgs())

            check("runs dir created", agent_mod.RUNS_DIR.exists())
            check("best dir created", agent_mod.BEST_DIR.exists())
            check("logs dir created", agent_mod.LOG_DIR.exists())
            check("state file created", agent_mod.STATE_FILE.exists())

            candidates = list(agent_mod.RUNS_DIR.glob("*.py"))
            check("candidate files written", len(candidates) == 2, f"got {len(candidates)}")

            state = json.loads(agent_mod.STATE_FILE.read_text())
            check("state step is 2", state["step"] == 2)
            check("state has history", len(state["history"]) == 2)

            log_files = list(agent_mod.LOG_DIR.glob("*.log"))
            check("log file created", len(log_files) >= 1)

        finally:
            agent_mod.RUNS_DIR = orig_runs
            agent_mod.BEST_DIR = orig_best
            agent_mod.STATE_FILE = orig_state
            agent_mod.LOG_DIR = orig_log
            os.environ.pop("CHUTES_API_KEY", None)


if __name__ == "__main__":
    print("=" * 60)
    print("  ARBOS AGENT — COMPREHENSIVE TESTS")
    print("=" * 60)

    test_hparams()
    test_validate_code()
    test_state_persistence()
    test_format_diagnostics()
    test_format_history()
    test_save_best()
    test_strip_thinking_tags()
    test_parse_response()
    test_llm_client_init()
    test_tester_payload()
    test_local_tester_payload()
    test_env_loading()
    test_prompt_file()
    test_full_dryrun_with_mock()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)
