"""Tests for configuration loading."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from crusades.config import HParams, StorageConfig


class TestHParamsLoad:
    """HParams loading from JSON."""

    def test_load_from_project_hparams(self):
        """Loads real hparams.json from repository."""
        hparams = HParams.load()
        assert hparams.netuid == 3
        assert hparams.evaluation_runs >= 1
        assert hparams.eval_timeout > 0
        assert hparams.benchmark_model_name != ""

    def test_load_from_explicit_path(self, tmp_path: Path):
        """Loads from an explicit path."""
        data = {
            "netuid": 99,
            "burn_rate": 0.05,
            "burn_uid": 0,
            "evaluation_runs": 2,
            "eval_steps": 20,
            "eval_timeout": 3600,
            "benchmark_model_name": "test/model",
            "benchmark_dataset_name": "test/data",
            "benchmark_dataset_split": "train",
            "benchmark_data_samples": 100,
            "benchmark_master_seed": 42,
            "benchmark_sequence_length": 1024,
            "benchmark_batch_size": 16,
            "set_weights_interval_blocks": 50,
            "reveal_blocks": 10,
            "min_blocks_between_commits": 5,
            "block_time": 12,
        }
        path = tmp_path / "hparams.json"
        path.write_text(json.dumps(data))

        hparams = HParams.load(path)
        assert hparams.netuid == 99
        assert hparams.burn_rate == 0.05

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            HParams.load(tmp_path / "nonexistent.json")

    def test_missing_required_fields_raises(self, tmp_path: Path):
        path = tmp_path / "hparams.json"
        path.write_text(json.dumps({"netuid": 1}))
        with pytest.raises(ValidationError):
            HParams.load(path)

    def test_burn_rate_bounds(self, tmp_path: Path):
        """burn_rate must be in [0.0, 1.0]."""
        data = {
            "netuid": 1,
            "burn_rate": 1.5,  # invalid
            "burn_uid": 0,
            "evaluation_runs": 2,
            "eval_steps": 20,
            "eval_timeout": 3600,
            "benchmark_model_name": "m",
            "benchmark_dataset_name": "d",
            "benchmark_dataset_split": "train",
            "benchmark_data_samples": 100,
            "benchmark_master_seed": 42,
            "benchmark_sequence_length": 1024,
            "benchmark_batch_size": 16,
            "set_weights_interval_blocks": 50,
            "reveal_blocks": 10,
            "min_blocks_between_commits": 5,
            "block_time": 12,
        }
        path = tmp_path / "hparams.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValidationError):
            HParams.load(path)


class TestNestedConfigs:
    """Nested config defaults."""

    def test_storage_config_default(self):
        sc = StorageConfig()
        assert "sqlite" in sc.database_url

    def test_mfu_config_from_hparams(self):
        hparams = HParams.load()
        assert hparams.mfu.gpu_peak_tflops > 0
        assert hparams.mfu.max_plausible_mfu > hparams.mfu.min_mfu

    def test_adaptive_threshold_config(self):
        hparams = HParams.load()
        assert hparams.adaptive_threshold.base_threshold > 0
        assert 0 < hparams.adaptive_threshold.decay_percent < 1
        assert hparams.adaptive_threshold.decay_interval_blocks > 0
