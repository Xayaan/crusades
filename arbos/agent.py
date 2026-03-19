"""Arbos Agent — Crusades MFU Optimization Loop.

Continuously improves train.py for maximum MFU using an LLM + testing backend.
Supports Basilica (cloud GPUs) or local Docker evaluation.
Logs all attempts, saves improvements, and leaves submission to the human.

Usage:
    # Basilica cloud (default)
    python arbos/agent.py --train-py local_test/train_fsdp.py

    # Local Docker (no cloud costs, uses your own GPUs)
    python arbos/agent.py --train-py local_test/train_fsdp.py --local

    # Local Docker with specific GPUs
    python arbos/agent.py --train-py local_test/train_fsdp.py --local --gpu-devices 4,5

    # Dry run (skip evaluation, test LLM generation only)
    python arbos/agent.py --train-py local_test/train_fsdp.py --dry-run

    # Limit steps
    python arbos/agent.py --train-py local_test/train_fsdp.py --max-steps 10
"""

import argparse
import difflib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ARBOS_DIR = Path(__file__).parent
PROJECT_ROOT = ARBOS_DIR.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
RUNS_DIR = ARBOS_DIR / "runs"
BEST_DIR = ARBOS_DIR / "best_submissions"
STATE_FILE = ARBOS_DIR / "state.json"
LOG_DIR = ARBOS_DIR / "logs"


def setup_logging() -> logging.Logger:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("arbos")
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    log_file = LOG_DIR / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


def compute_code_diff(old_code: str, new_code: str) -> str:
    """Return a unified diff showing changed lines."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(old_lines, new_lines, fromfile="best.py", tofile="candidate.py", n=2)
    )
    if not diff:
        return "(no changes)"
    meaningful = [line.rstrip("\n") for line in diff[2:] if line.startswith(("+", "-", "@"))]
    return "\n".join(meaningful)


@dataclass
class Attempt:
    step: int
    reasoning: str
    mfu: float
    success: bool
    error: str | None = None
    error_code: str | None = None
    diagnostics_text: str | None = None
    code_diff: str | None = None
    candidate_code: str | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class AgentState:
    best_mfu: float = 0.0
    best_code_path: str = ""
    step: int = 0
    total_improvements: int = 0
    history: list[dict] = field(default_factory=list)

    def add_attempt(self, attempt: Attempt):
        self.history.append(asdict(attempt))
        if len(self.history) > 50:
            self.history = self.history[-50:]
        # Only keep candidate_code on the most recent failed entry to limit state size
        for h in self.history[:-1]:
            h.pop("candidate_code", None)

    def save(self):
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "AgentState":
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                return cls(**valid)
            except (json.JSONDecodeError, TypeError):
                pass
        return cls()


def load_hparams() -> dict:
    path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    if not path.exists():
        raise FileNotFoundError(f"hparams.json not found at {path}")
    return json.loads(path.read_text())


def validate_code(code: str) -> tuple[bool, str]:
    if "def inner_steps" not in code:
        return False, "Missing 'def inner_steps' function"
    if "def get_strategy" not in code:
        return False, "Missing 'def get_strategy' function"
    if "InnerStepsResult" not in code:
        return False, "Missing InnerStepsResult"
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    try:
        from security_scanner import validate_code_security

        ok, msg = validate_code_security(code)
        if not ok:
            return False, msg
    except ImportError:
        pass

    return True, "OK"


def save_best(code: str, step: int, mfu: float) -> Path:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"train_v{step}_{mfu:.1f}.py"
    path = BEST_DIR / filename
    path.write_text(code)
    latest = BEST_DIR / "train_latest.py"
    latest.write_text(code)
    return path


def format_diagnostics(diag: dict) -> str:
    """Format Basilica diagnostics dict into readable text for the LLM."""
    if not diag:
        return ""
    parts = []

    verification = diag.get("verification", {})
    if verification:
        failed = verification.get("checks_failed", [])
        if failed:
            for check in failed:
                name = check.get("check", "unknown")
                err = check.get("error", "")
                parts.append(f"  check_failed: {name} — {err}")
        passed = verification.get("checks_passed", [])
        if passed:
            parts.append(f"  checks_passed: {', '.join(str(c) for c in passed)}")
        grad = verification.get("gradient_verification", {})
        if grad:
            for k, v in grad.items():
                parts.append(f"  gradient_{k}: {v}")
        weight = verification.get("weight_verification", {})
        if weight:
            for k, v in weight.items():
                parts.append(f"  weight_{k}: {v}")

    for key in (
        "reference_loss",
        "candidate_loss",
        "strategy",
        "expected_tokens",
        "actual_tokens",
        "total_unique_tokens",
        "model_params",
        "gpu_peak_tflops",
    ):
        if key in diag:
            parts.append(f"  {key}: {diag[key]}")

    params_changed = diag.get("params_changed", {})
    if params_changed:
        for k, v in params_changed.items():
            parts.append(f"  params_changed_{k}: {v}")

    trainable = diag.get("trainable_params", {})
    if trainable:
        for k, v in trainable.items():
            parts.append(f"  trainable_{k}: {v}")

    return "\n".join(parts)


def format_history(history: list[dict], best_mfu: float = 0.0, max_entries: int = 15) -> str:
    if not history:
        return "No previous attempts."
    recent = history[-max_entries:]
    lines = []
    running_best = 0.0
    for h in recent:
        mfu_val = h.get("mfu", 0)
        is_success = h.get("success", False)
        if is_success and mfu_val > running_best:
            status = "IMPROVED"
            running_best = mfu_val
        elif is_success and mfu_val > 0:
            status = "OK"
        else:
            status = "FAIL"
        mfu_str = f"{mfu_val:.2f}%" if mfu_val > 0 else "N/A"
        entry = f"### Step {h['step']}: [{status}] MFU={mfu_str}\n"
        entry += f"Strategy: {h['reasoning']}\n"

        if h.get("error_code"):
            entry += f"Error code: {h['error_code']}\n"

        if h.get("error"):
            entry += f"Error: {h['error']}\n"

        if h.get("diagnostics_text"):
            entry += f"Diagnostics:\n{h['diagnostics_text']}\n"

        if h.get("code_diff") and h["code_diff"] != "(no changes)":
            entry += f"Changes made:\n```\n{h['code_diff']}\n```\n"

        lines.append(entry)

    # Include the full failing code from the most recent failed attempt so the
    # LLM can see exactly what broke (diffs alone are hard to reason about for
    # strategy switches like FSDP → DDP).
    last = recent[-1]
    if not last.get("success") and last.get("candidate_code"):
        lines.append(
            f"\n### MOST RECENT FAILING CODE (Step {last['step']}):\n"
            f"```python\n{last['candidate_code']}\n```\n"
        )

    return "\n".join(lines)


def print_summary_table(history: list[dict], log: logging.Logger):
    improvements = [h for h in history if h.get("success") and h.get("mfu", 0) > 0]
    if not improvements:
        return
    log.info("")
    log.info("IMPROVEMENT HISTORY")
    log.info("-" * 70)
    log.info(f" {'#':>3} | {'MFU':>7} | {'Step':>5} | Change")
    log.info("-" * 70)

    best_so_far = 0.0
    idx = 0
    for h in improvements:
        if h["mfu"] > best_so_far:
            idx += 1
            delta = f"+{h['mfu'] - best_so_far:.1f}%" if best_so_far > 0 else "base"
            best_so_far = h["mfu"]
            log.info(
                f" {idx:>3} | {h['mfu']:>6.1f}% | {h['step']:>5} | {delta:>6} | {h['reasoning'][:40]}"
            )
    log.info("-" * 70)


def run_agent(args):
    log = setup_logging()

    if args.env_file:
        env_path = Path(args.env_file)
    else:
        env_path = ARBOS_DIR / ".env"

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    initial_path = Path(args.train_py)
    if not initial_path.exists():
        log.error(f"File not found: {initial_path}")
        sys.exit(1)

    initial_code = initial_path.read_text()
    valid, msg = validate_code(initial_code)
    if not valid:
        log.error(f"Invalid initial train.py: {msg}")
        sys.exit(1)

    hparams = load_hparams()

    from llm_client import LLMClient
    from tester import BasilicaTester, LocalDockerTester

    llm = LLMClient()
    tester = None
    if not args.dry_run:
        if args.local:
            tester = LocalDockerTester(
                hparams,
                project_root=PROJECT_ROOT,
                image=args.docker_image,
                num_gpus=args.num_gpus,
                gpu_devices=args.gpu_devices,
            )
        else:
            tester = BasilicaTester(hparams)

    state = AgentState.load()
    mode = args.mode

    backend = "dry-run" if args.dry_run else ("local-docker" if args.local else "basilica")

    log.info("")
    log.info("=" * 60)
    log.info("  ARBOS — Crusades MFU Optimization Agent")
    log.info("=" * 60)
    log.info(f"  Mode:          {mode}")
    log.info(f"  Backend:       {backend}")
    log.info(f"  LLM:           {llm.provider} ({llm.model})")
    log.info(f"  Initial file:  {initial_path}")
    log.info(f"  Dry run:       {args.dry_run}")
    if state.best_mfu > 0:
        log.info(f"  Resuming:      step {state.step}, best MFU {state.best_mfu:.1f}%")
    log.info("=" * 60)
    log.info("")

    if not state.best_code_path or not Path(state.best_code_path).exists():
        if tester:
            log.info(f"Testing initial submission ({backend})...")
            result = tester.evaluate(initial_code)
            if result.success:
                state.best_mfu = result.mfu
                log.info(f"Initial MFU: {result.mfu:.2f}%")
                state.add_attempt(
                    Attempt(
                        step=0,
                        reasoning="Initial baseline submission",
                        mfu=result.mfu,
                        success=True,
                    )
                )
            else:
                log.warning(f"Initial test failed: {result.error}")
                if result.error_code:
                    log.warning(f"Error code: {result.error_code}")
                if result.diagnostics:
                    log.debug(f"Diagnostics: {result.diagnostics}")
                log.info("Starting with MFU=0 (will accept any passing result)")
                state.add_attempt(
                    Attempt(
                        step=0,
                        reasoning="Initial baseline submission",
                        mfu=0.0,
                        success=False,
                        error=result.error,
                        error_code=result.error_code,
                        diagnostics_text=format_diagnostics(result.diagnostics),
                    )
                )
        else:
            log.info("[DRY RUN] Skipping initial evaluation")

        path = save_best(initial_code, 0, state.best_mfu)
        state.best_code_path = str(path)
        state.save()

    best_code = Path(state.best_code_path).read_text()
    consecutive_failures = 0

    try:
        while True:
            if args.max_steps and state.step >= args.max_steps:
                log.info(f"Reached max steps ({args.max_steps}). Stopping.")
                break

            state.step += 1
            step = state.step

            log.info("")
            log.info(f"{'=' * 60}")
            log.info(f"  STEP {step}")
            log.info(f"{'=' * 60}")

            if consecutive_failures > 0:
                delay = min(2**consecutive_failures, 120)
                log.info(f"Backoff: {delay}s after {consecutive_failures} consecutive failure(s)")
                time.sleep(delay)

            try:
                response = llm.suggest_improvement(
                    code=best_code,
                    mfu=state.best_mfu,
                    history=format_history(state.history, best_mfu=state.best_mfu),
                    hparams=hparams,
                )
            except Exception as e:
                log.error(f"LLM error: {e}")
                consecutive_failures += 1
                state.add_attempt(
                    Attempt(
                        step=step, reasoning=f"LLM error: {e}", mfu=0.0, success=False, error=str(e)
                    )
                )
                state.save()
                continue

            log.info(f"Strategy: {response.reasoning}")

            diff = compute_code_diff(best_code, response.code)

            valid, msg = validate_code(response.code)
            if not valid:
                log.warning(f"Invalid code: {msg}")
                error_code = (
                    "security_violation" if "Security violation" in msg else "validation_error"
                )
                consecutive_failures += 1
                state.add_attempt(
                    Attempt(
                        step=step,
                        reasoning=response.reasoning,
                        mfu=0.0,
                        success=False,
                        error=msg,
                        error_code=error_code,
                        code_diff=diff,
                        candidate_code=response.code,
                    )
                )
                state.save()
                continue
            candidate_path = RUNS_DIR / f"step_{step:04d}_candidate.py"
            candidate_path.write_text(response.code)

            if args.dry_run:
                log.info(f"[DRY RUN] Candidate saved to: {candidate_path}")
                state.add_attempt(
                    Attempt(
                        step=step,
                        reasoning=response.reasoning,
                        mfu=0.0,
                        success=True,
                        code_diff=diff,
                    )
                )
                state.save()
                continue

            log.info(f"Sending for evaluation ({backend})...")
            try:
                result = tester.evaluate(response.code)
            except Exception as e:
                log.error(f"Evaluation error: {e}")
                consecutive_failures += 1
                state.add_attempt(
                    Attempt(
                        step=step,
                        reasoning=response.reasoning,
                        mfu=0.0,
                        success=False,
                        error=str(e),
                        code_diff=diff,
                        candidate_code=response.code,
                    )
                )
                state.save()
                continue

            if not result.success:
                log.warning(f"Evaluation FAILED: {result.error}")
                if result.error_code:
                    log.warning(f"Error code: {result.error_code}")
                if result.diagnostics:
                    log.debug(f"Diagnostics: {result.diagnostics}")

                diag_text = format_diagnostics(result.diagnostics)

                consecutive_failures += 1
                state.add_attempt(
                    Attempt(
                        step=step,
                        reasoning=response.reasoning,
                        mfu=result.mfu,
                        success=False,
                        error=result.error,
                        error_code=result.error_code,
                        diagnostics_text=diag_text,
                        code_diff=diff,
                        candidate_code=response.code,
                    )
                )
                state.save()
                continue

            consecutive_failures = 0
            log.info(
                f"Result: MFU={result.mfu:.2f}% | TPS={result.tps:.1f} | "
                f"Tokens={result.total_tokens} | Time={result.wall_time:.1f}s"
            )

            if result.mfu > state.best_mfu:
                improvement = result.mfu - state.best_mfu
                state.total_improvements += 1

                log.info("")
                log.info("*" * 60)
                log.info(f"  NEW BEST MFU: {result.mfu:.2f}% (+{improvement:.2f}%)")
                log.info(f"  Improvement #{state.total_improvements} at step {step}")
                log.info(f"  Change: {response.reasoning}")
                log.info("*" * 60)

                path = save_best(response.code, step, result.mfu)
                state.best_mfu = result.mfu
                state.best_code_path = str(path)
                best_code = response.code

                log.info(f"  Saved to: {path}")
                log.info("")
                log.info("  READY FOR SUBMISSION")
                log.info(f"  File: {path}")
                if mode == "validator":
                    log.info("  Submit from burn_uid 3 (no submission fee):")
                else:
                    log.info("  Submit (0.05 TAO fee):")
                log.info("    1. Host the file at a URL (e.g. GitHub Gist)")
                log.info(
                    "    2. uv run -m neurons.miner submit <url> "
                    "--wallet.name <name> --wallet.hotkey <hotkey>"
                )
                log.info("")
            else:
                log.info(f"No improvement: {result.mfu:.2f}% (best: {state.best_mfu:.2f}%)")

            state.add_attempt(
                Attempt(
                    step=step,
                    reasoning=response.reasoning,
                    mfu=result.mfu,
                    success=True,
                    code_diff=diff,
                    diagnostics_text=f"  tps: {result.tps:.1f}\n  wall_time: {result.wall_time:.1f}s\n  tokens: {result.total_tokens}",
                )
            )
            state.save()

    except KeyboardInterrupt:
        log.info("\nStopped by user.")
    finally:
        log.info("")
        log.info("=" * 60)
        log.info("  SESSION SUMMARY")
        log.info("=" * 60)
        log.info(f"  Total steps:        {state.step}")
        log.info(f"  Total improvements: {state.total_improvements}")
        log.info(f"  Best MFU:           {state.best_mfu:.2f}%")
        log.info(f"  Best submission:    {state.best_code_path}")
        log.info("=" * 60)

        print_summary_table(state.history, log)

        if tester:
            tester.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Arbos — Crusades MFU Optimization Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Start optimizing from reference FSDP implementation (Basilica cloud)
  python arbos/agent.py --train-py local_test/train_fsdp.py

  # Use local Docker instead of Basilica (no cloud costs)
  python arbos/agent.py --train-py local_test/train_fsdp.py --local

  # Local Docker with specific GPUs
  python arbos/agent.py --train-py local_test/train_fsdp.py --local --gpu-devices 4,5

  # Local Docker with custom image and GPU count
  python arbos/agent.py --train-py local_test/train_fsdp.py --local --num-gpus 4

  # Run as validator team (burn_uid 3)
  python arbos/agent.py --train-py local_test/train_fsdp.py --mode validator

  # Test LLM code generation without evaluation (no GPU costs)
  python arbos/agent.py --train-py local_test/train_fsdp.py --dry-run

  # Run 20 optimization steps then stop
  python arbos/agent.py --train-py local_test/train_fsdp.py --max-steps 20
""",
    )
    parser.add_argument(
        "--train-py",
        required=True,
        help="Path to initial train.py file to optimize",
    )
    parser.add_argument(
        "--mode",
        choices=["miner", "validator"],
        default="miner",
        help="Running mode: miner (pays fee) or validator (burn_uid, no fee)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Docker evaluation instead of Basilica cloud",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for local Docker (default: docker.num_gpus from hparams)",
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default=None,
        help="Specific GPU device IDs for local Docker, e.g. '4,5' or '0,1,2,3'",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=None,
        help="Docker image for local evaluation (default: from hparams or templar-eval:latest)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max optimization steps (runs forever if not set)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip evaluation (test LLM generation only)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file (default: arbos/.env)",
    )
    args = parser.parse_args()

    local_only_flags = []
    if args.num_gpus is not None:
        local_only_flags.append("--num-gpus")
    if args.gpu_devices is not None:
        local_only_flags.append("--gpu-devices")
    if args.docker_image is not None:
        local_only_flags.append("--docker-image")
    if local_only_flags and not args.local:
        parser.error(
            f"{', '.join(local_only_flags)} require --local (otherwise Basilica is used "
            f"and these flags are ignored)"
        )

    run_agent(args)


if __name__ == "__main__":
    main()
