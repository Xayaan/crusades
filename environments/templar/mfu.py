"""Model FLOPs Utilization calculation."""


def calculate_mfu(
    total_unique_tokens: int,
    wall_time: float,
    model_params: int,
    gpu_peak_tflops: float = 312.0,
    num_gpus: int = 1,
) -> float:
    """Calculate system-level Model FLOPs Utilization (MFU).

    MFU = total_system_flops / total_system_peak_flops

    The caller provides *total_unique_tokens* -- the number of distinct tokens
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
