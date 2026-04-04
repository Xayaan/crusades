"""Torchrun subprocess management for multi-GPU evaluation."""

import asyncio
import logging
import os
import random

logger = logging.getLogger(__name__)

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


async def evaluate_via_torchrun(request) -> dict:
    """Spawn torchrun for multi-GPU Basilica evaluation (uvicorn is single-process).

    Args:
        request: An EvaluateRequest instance (or any object with matching attributes
                 and a model_dump() method).
    """
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
                return _json.loads(line[len("EVAL_RESULT:"):])

        return {
            "task_id": request.task_id,
            "success": False,
            "error": (
                f"No EVAL_RESULT in torchrun output "
                f"(exit {proc.returncode}): "
                f"{stdout_text[-1000:]}"
            ),
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
