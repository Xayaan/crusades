#!/usr/bin/env python3
"""
Setup benchmark data and model for local miner testing.

This script downloads and caches:
1. The benchmark model defined in hparams.json "benchmark_model_name"
2. Pre-tokenized data samples defined in hparams.json "benchmark_dataset_name"
3. Number of samples defined in hparams.json "benchmark_data_samples"
4. Sequence length defined in hparams.json "benchmark_sequence_length"
5. Master seed defined in hparams.json "benchmark_master_seed"

Run once before testing your train.py locally:
    uv run local_test/setup_benchmark.py
"""

import itertools
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_hparams():
    """Load hparams.json configuration."""
    hparams_path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    with open(hparams_path) as f:
        return json.load(f)


def setup_model(model_name: str, output_dir: Path):
    """Download and cache the benchmark model."""
    print(f"Downloading model: {model_name}")
    print(f"   Output: {output_dir}")

    # Download tokenizer
    print("   Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Download model
    print("   Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.save_pretrained(output_dir)

    print(f"   Model saved to: {output_dir}")


def setup_data(
    model_name: str,
    dataset_name: str,
    num_samples: int,
    sequence_length: int,
    seed: int,
    output_path: Path,
):
    """Download and tokenize benchmark data."""
    print("Setting up benchmark data:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Samples: {num_samples:,}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Seed: {seed}")
    print(f"   Output: {output_path}")

    # Load tokenizer
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset with streaming
    print("   Loading dataset (streaming)...")
    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )

    # Shuffle with buffer_size for efficiency
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    # Collect texts first, then batch tokenize
    print(f"   Collecting {num_samples:,} samples...")
    texts = []
    dataset_iter = iter(dataset)

    for i, sample in enumerate(itertools.islice(dataset_iter, num_samples)):
        if i % 10000 == 0:
            print(f"      Progress: {i:,}/{num_samples:,}")
        text = sample.get("text", sample.get("content", ""))
        if text:
            texts.append(text)

    # Batch tokenize (much faster than one-by-one)
    print(f"   Batch tokenizing {len(texts):,} samples...")
    encoded = tokenizer(
        texts,
        max_length=sequence_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    data = encoded["input_ids"]
    print(f"   Data shape: {data.shape}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    print(f"   Saved to: {output_path}")

    return data


def main():
    print("=" * 60)
    print("BENCHMARK SETUP")
    print("=" * 60)
    print()

    # Load config
    hparams = load_hparams()
    model_name = hparams.get("benchmark_model_name", "Qwen/Qwen2.5-7B")
    dataset_name = hparams.get("benchmark_dataset_name", "HuggingFaceFW/fineweb")
    num_samples = hparams.get("benchmark_data_samples", 10000)
    sequence_length = hparams.get("benchmark_sequence_length", 1024)
    seed = hparams.get("benchmark_master_seed", 42)

    print("Configuration from hparams.json:")
    print(f"   Model: {model_name}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Samples: {num_samples:,}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Seed: {seed}")
    print()

    # Paths
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"

    # Setup model
    if model_dir.exists():
        print(f"Model already exists at {model_dir}")
        print("   Delete it to re-download")
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        setup_model(model_name, model_dir)
    print()

    # Setup data
    if data_path.exists():
        print(f"Data already exists at {data_path}")
        print("   Delete it to regenerate")
    else:
        setup_data(model_name, dataset_name, num_samples, sequence_length, seed, data_path)
    print()

    print("=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print()
    print("To verify your train.py, run the Docker-based simulator:")
    print()
    print("  See local_test/simulate_validator.py for usage instructions.")
    print()


if __name__ == "__main__":
    main()
