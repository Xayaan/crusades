"""
Local Validator Simulation Tool.

Test your train.py exactly as the production validator does, inside the
same Docker container.

1. Build the docker image (run from repo root):

   docker build --network=host --no-cache -f environments/templar/Dockerfile -t templar-eval:latest .

2. Run the simulation (requires 4x A100 GPUs for 7B model):

    # Using GPUs 4,5,6,7 (adjust --gpus flag for your device IDs):
    docker run --gpus '"device=4,5,6,7"' -it --rm \
        --ipc=host \
        --ulimit memlock=-1:-1 \
        -e NCCL_P2P_LEVEL=NVL \
        -e NCCL_SHM_USE_CUDA_MEMCPY=1 \
        -e NCCL_NVLS_ENABLE=1 \
        -e NCCL_IB_DISABLE=1 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -v "$(pwd)/local_test/train_mixed.py":/test/train.py:ro \
        -v "$(pwd)/local_test/simulate_validator.py":/test/simulate.py:ro \
        -v "$(pwd)/hparams/hparams.json":/app/hparams.json:ro \
        -v "$(pwd)/environments/templar/env.py":/app/env.py:ro \
        -v "$(pwd)/src/crusades/core/security_defs.py":/app/crusades/core/security_defs.py:ro \
        -e PYTHONPATH=/app \
        templar-eval:latest \
        python3 /test/simulate.py

    # Using all GPUs (if only 4 available):
    docker run --gpus 4 -it --rm \
        --ipc=host \
        --ulimit memlock=-1:-1 \
        -e NCCL_P2P_LEVEL=NVL \
        -e NCCL_SHM_USE_CUDA_MEMCPY=1 \
        -e NCCL_NVLS_ENABLE=1 \
        -e NCCL_IB_DISABLE=1 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -v "$(pwd)/local_test/train_mixed.py":/test/train.py:ro \
        -v "$(pwd)/local_test/simulate_validator.py":/test/simulate.py:ro \
        -v "$(pwd)/hparams/hparams.json":/app/hparams.json:ro \
        -v "$(pwd)/environments/templar/env.py":/app/env.py:ro \
        -v "$(pwd)/src/crusades/core/security_defs.py":/app/crusades/core/security_defs.py:ro \
        -e PYTHONPATH=/app \
        templar-eval:latest \
        python3 /test/simulate.py

   Strategies for 4 GPUs:
   - train_mixed.py: dp_size=2, tp_size=2 (mixed DP+TP)
   - train_ddp.py:   dp_size=4, tp_size=1 (4-way DDP) — update get_strategy()
   - train_fsdp.py:  dp_size=4, tp_size=1 (4-way FSDP) — update get_strategy()
   - train_tp.py:    dp_size=1, tp_size=4 (4-way TP) — update get_strategy()
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.append("/app")


def _sanitize(obj):
    """Make diagnostics JSON-serializable (handles tensors etc.)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "detach"):
        return obj.cpu().detach().tolist()
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj


def _load_config():
    hparams_path = Path("/app/hparams.json")
    if not hparams_path.exists():
        hparams_path = Path("/app/hparams/hparams.json")
    if not hparams_path.exists():
        print("ERROR: hparams.json not found. See docstring for launch command.")
        sys.exit(1)
    print(f"Loading hparams from: {hparams_path}")
    with open(hparams_path) as f:
        return json.load(f)


def _build_payload(hb, code):
    return {
        "task_id": 1337,
        "seed": "local:test:1",
        "model_url": hb["benchmark_model_name"],
        "data_url": hb["benchmark_dataset_name"],
        "steps": hb["eval_steps"],
        "batch_size": hb["benchmark_batch_size"],
        "sequence_length": hb["benchmark_sequence_length"],
        "data_samples": hb["benchmark_data_samples"],
        "timeout": hb["eval_timeout"],
        "code": code,
        "use_random_init": True,
        "min_trainable_params_ratio": 1.0,
        "max_loss_difference": hb["verification"]["max_loss_difference"],
        "min_params_changed_ratio": hb["verification"]["min_params_changed_ratio"],
        "weight_relative_error_max": hb["verification"]["weight_relative_error_max"],
        "timer_divergence_threshold": hb["verification"]["timer_divergence_threshold"],
        "gpu_peak_tflops": hb["mfu"]["gpu_peak_tflops"],
        "max_plausible_mfu": hb["mfu"]["max_plausible_mfu"],
        "min_mfu": hb["mfu"]["min_mfu"],
        "num_gpus": hb.get("docker", {}).get("num_gpus", 1),
    }


def _print_result(result):
    print()
    print("=" * 50)
    print("VALIDATOR RESULT")
    print("=" * 50)
    print(f"Success: {result.get('success')}")
    if not result.get("success"):
        print(f"Error: {result.get('error')}")
        print(f"Error Code: {result.get('error_code')}")
    print(f"MFU: {result.get('mfu', 0.0):.2f}%")
    print(f"TPS: {result.get('tps', 0.0):.2f}")
    print()
    print("Diagnostics:")
    print(json.dumps(_sanitize(result.get("diagnostics", {})), indent=2))
    print("=" * 50)


async def _simulate_single_gpu(payload):
    """Direct Actor call — single process, no torchrun needed."""
    from env import Actor

    actor = Actor()
    return await actor.evaluate(
        task_id=payload["task_id"],
        seed=payload["seed"],
        model_url=payload["model_url"],
        data_url=payload["data_url"],
        steps=payload["steps"],
        batch_size=payload["batch_size"],
        sequence_length=payload["sequence_length"],
        data_samples=payload["data_samples"],
        timeout=payload["timeout"],
        code=payload["code"],
        use_random_init=payload["use_random_init"],
        min_trainable_params_ratio=payload["min_trainable_params_ratio"],
        max_loss_difference=payload["max_loss_difference"],
        min_params_changed_ratio=payload["min_params_changed_ratio"],
        weight_relative_error_max=payload["weight_relative_error_max"],
        timer_divergence_threshold=payload["timer_divergence_threshold"],
        gpu_peak_tflops=payload["gpu_peak_tflops"],
        max_plausible_mfu=payload["max_plausible_mfu"],
        min_mfu=payload["min_mfu"],
        require_cuda_timing=True,
        num_gpus=1,
    )


async def _simulate_multi_gpu(payload):
    """Spawn torchrun and stream output so logs are visible (warmup, timing, etc.)."""
    import tempfile

    num_gpus = payload["num_gpus"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir="/tmp", delete=False) as f:
        json.dump(payload, f)
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

    proc = await asyncio.create_subprocess_exec(
        "torchrun",
        "--nproc_per_node",
        str(num_gpus),
        script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    result = None
    async for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").rstrip()
        if line.startswith("EVAL_RESULT:"):
            result = json.loads(line[len("EVAL_RESULT:") :])
        else:
            print(line)

    await proc.wait()

    for p in (params_path, script_path):
        try:
            import os

            os.unlink(p)
        except OSError:
            pass

    if result is None:
        return {
            "task_id": payload["task_id"],
            "success": False,
            "error": f"No EVAL_RESULT in torchrun output (exit {proc.returncode})",
            "mfu": 0.0,
            "tps": 0.0,
            "total_tokens": 0,
            "wall_time_seconds": 0.0,
            "seed": payload["seed"],
        }
    return result


async def simulate():
    print("Starting Simulated Validator Test...")

    hb = _load_config()

    code_path = Path("/test/train.py")
    if not code_path.exists():
        print("ERROR: train.py not found at /test/train.py. See docstring for launch command.")
        sys.exit(1)

    print(f"Loading miner code from: {code_path}")
    code = code_path.read_text()

    payload = _build_payload(hb, code)
    num_gpus = payload["num_gpus"]
    print(f"num_gpus={num_gpus}")

    if num_gpus > 1:
        print(f"Multi-GPU mode: spawning torchrun with {num_gpus} GPUs")
        result = await _simulate_multi_gpu(payload)
    else:
        print("Single-GPU mode: direct Actor.evaluate()")
        result = await _simulate_single_gpu(payload)

    _print_result(result)


if __name__ == "__main__":
    asyncio.run(simulate())
