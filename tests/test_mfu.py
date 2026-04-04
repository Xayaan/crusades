"""Tests for MFU calculation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environments', 'templar'))
from mfu import calculate_mfu


class TestMFUCalculation:
    def test_basic_mfu(self):
        # 1B params, 1M tokens, 10s, single A100
        mfu = calculate_mfu(
            total_unique_tokens=1_000_000,
            wall_time=10.0,
            model_params=1_000_000_000,
            gpu_peak_tflops=312.0,
            num_gpus=1,
        )
        # 6 * 1B * 1M = 6e15 FLOPS
        # 312 * 1e12 * 10 = 3.12e15 peak
        # MFU = 6e15 / 3.12e15 * 100 ~ 192% -> capped at 100
        assert mfu == 100.0

    def test_zero_wall_time(self):
        mfu = calculate_mfu(0, 0.0, 1_000_000)
        assert mfu == 0.0

    def test_zero_gpu(self):
        mfu = calculate_mfu(1000, 10.0, 1_000_000, num_gpus=0)
        assert mfu == 0.0

    def test_multi_gpu(self):
        mfu_single = calculate_mfu(1000, 10.0, 1_000_000, gpu_peak_tflops=312.0, num_gpus=1)
        mfu_multi = calculate_mfu(1000, 10.0, 1_000_000, gpu_peak_tflops=312.0, num_gpus=2)
        # Same work spread over more peak = lower MFU
        assert mfu_multi < mfu_single

    def test_negative_wall_time(self):
        mfu = calculate_mfu(1000, -5.0, 1_000_000)
        assert mfu == 0.0

    def test_reasonable_mfu(self):
        # Realistic scenario: 7B params, 10k tokens, 60s, 4 GPUs
        mfu = calculate_mfu(
            total_unique_tokens=10_000,
            wall_time=60.0,
            model_params=7_000_000_000,
            gpu_peak_tflops=312.0,
            num_gpus=4,
        )
        assert 0.0 < mfu <= 100.0
