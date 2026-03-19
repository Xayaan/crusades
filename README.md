# Templar Crusades

**MFU Crusades on Bittensor** - Miners compete to optimize training code for maximum MFU (Model FLOPs Utilization).

## How It Works

```
┌───────────────────────────────────────────────────────────────────────────────── ┐
│                              Crusades FLOW                                       │
│                                                                                  │
│   MINER                        BLOCKCHAIN                      VALIDATOR         │
│     │                                                              │             │
│     │  1. Host train.py at URL                                     │             │
│     │     (Gist, Pastebin, etc)                                    │             │
│     │                                                              │             │
│     ├──▶ 2. Submit URL ─────────▶ set_reveal_commitment            │             │
│     │                             (timelock encrypted)             │             │
│     │                                    │                         │             │
│     │                                    │ (wait reveal_blocks)    │             │
│     │                                    ▼                         │             │
│     │                              3. Decrypted ◀───────────────── ┤ Read        │
│     │                                                              │             │
│     │                                                     4. Download code       │
│     │                                                        from URL            │
│     │                                                              │             │
│     │                                                     5. Runs in Container   │
│     │                                                        (X eval runs)       │
│     │                                                              │             │
│     │                                                     6. Calculate MFU       │
│     │                                                        (median score)      │
│     │                                                              │             │
│     │                                                     7. Set weights         │
│                                                                                  │
└───────────────────────────────────────────────────────────────────────────────── ┘
```

## Quick Start

### Prerequisites

```bash
# Clone and setup
git clone https://github.com/one-covenant/crusades
cd crusades
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Create .env (for HuggingFace access)
echo "HF_TOKEN=hf_your_token" > .env
```

---

## For Miners

### 1. Setup & Test Locally

```bash
# Download model & data for local testing
uv run local_test/setup_benchmark.py
```

> **Hardware requirement:** The benchmark uses 2x A100 80 GB GPUs. The model (Qwen2.5-3B) fits on a single GPU, so DDP, FSDP, and TP strategies are all viable.

### Simulate the Validator (Recommended)

Test your `train.py` inside the **exact same Docker container** the production validator uses. This gives MFU numbers that closely match the tournament leaderboard and avoids environment dependency mismatches.

```bash
# Build the eval image (from repo root)
docker build --network=host -f environments/templar/Dockerfile \
    --no-cache -t templar-eval:latest .

# Run the simulation (requires 2x A100 GPUs)
# Note: MFU/benchmark results will only closely match leaderboard numbers
# when executed on the same GPU model (A100). Runs on other hardware may
# produce divergent MFU results.
docker run --gpus 2 -it --rm \
    --ipc=host \
    --ulimit memlock=-1:-1 \
    -e NCCL_P2P_LEVEL=NVL \
    -e NCCL_SHM_USE_CUDA_MEMCPY=1 \
    -e NCCL_NVLS_ENABLE=1 \
    -e NCCL_IB_DISABLE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$(pwd)/local_test/train_fsdp.py":/test/train.py:ro \
    -v "$(pwd)/local_test/simulate_validator.py":/test/simulate.py:ro \
    -v "$(pwd)/hparams/hparams.json":/app/hparams.json:ro \
    -v "$(pwd)/environments/templar/env.py":/app/env.py:ro \
    -v "$(pwd)/src/crusades/core/security_defs.py":/app/crusades/core/security_defs.py:ro \
    -e PYTHONPATH=/app \
    templar-eval:latest \
    python3 /test/simulate.py
```

Replace `train_fsdp.py` with `train_ddp.py`, `train_tp.py`, or `train_mixed.py` to test other strategies. `train_mixed.py` requires 4 GPUs (`dp_size=2, tp_size=2`).

This runs the full evaluation pipeline: security scan, reference baseline, warmup, timed eval, weight verification, and MFU calculation — all with the same thresholds as production.

### 2. Host Your Code

Host your `train.py` at any URL that returns raw code:
- **GitHub Gist** (recommended - use secret gist for privacy)
- **Raw GitHub file** (use raw.githubusercontent.com)
- **Pastebin** or any paste service
- Any HTTP/HTTPS URL

### 3. Submit to Crusades

```bash

# Submit to mainnet
uv run -m neurons.miner submit "https://gist.github.com/user/gist_id" \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --network finney

# Submit to localnet (testing)
uv run -m neurons.miner submit "https://gist.github.com/user/gist_id" \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --network local
```

**Parameters**: `--wallet.name`, `--wallet.hotkey`, `--network` (finney/test/local)

---

## For Validators

See [docs/Validator.md](docs/Validator.md) for detailed validator setup.

---

## train.py Requirements

You submit a **single `train.py` file**. The files in `local_test/` (`train_fsdp.py`, `train_tp.py`, `train_ddp.py`, `train_mixed.py`, `train.py`) are reference examples only — use them as a starting point for your own `train.py`.

Your `train.py` must contain:
1. **`inner_steps()`** — the training function (required)
2. **`get_strategy()`** — declares your parallelism strategy (required for multi-GPU)

### `get_strategy()`

For multi-GPU submissions, define this function to declare your parallelism topology:

```python
def get_strategy():
    return {"dp_size": 2, "tp_size": 1}  # 2-way data parallel, no tensor parallel
```

The validator requires `dp_size * tp_size == num_gpus`. Common configurations:

| Topology | Data Handling | Description |
|----------|---------------|-------------|
| `{"dp_size": N, "tp_size": 1}` | Sharded (each DP group gets different batches) | Pure data parallel (DDP/FSDP) |
| `{"dp_size": 1, "tp_size": N}` | Replicated (all ranks get the same batches) | Pure tensor parallel |
| `{"dp_size": D, "tp_size": T}` | Sharded across DP groups, replicated within TP groups | Mixed parallelism (e.g., FSDP+TP) |

Legacy string values (`"ddp"`, `"fsdp"`, `"tp"`) are still accepted but the dict format is preferred.

**Important:** `get_strategy()` must return a literal value (string or dict). The validator parses this function via AST and only accepts direct `return` literals. Returning a computed value through a variable or helper function will be rejected even if the value itself is valid.

This matters because the validator uses `dp_size` and `tp_size` to decide data distribution and MFU token counting. Declaring the wrong topology will cause weight verification failures.

### `inner_steps()` Signature

```python
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    ...
    return InnerStepsResult(final_logits=..., total_tokens=..., final_loss=...)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `torch.nn.Module` | Pre-loaded model on the correct device, in train mode |
| `data_iterator` | `Iterator[torch.Tensor]` | Infinite iterator yielding `(batch_size, seq_len)` tensors. Data distribution follows the declared `dp_size`/`tp_size` topology. |
| `optimizer` | `None` | Always `None` — miner must create their own optimizer. |
| `num_steps` | `int` | Number of training steps to complete |
| `device` | `torch.device` | Target device (`cuda:0`, `cuda:1`, etc.). Use `device.index` for local rank. |
| `num_gpus` | `int` | Number of GPUs. `1` = single-GPU, `>1` = multi-GPU with `torchrun` (process group already initialized). |

### Return Value

`inner_steps` must return an `InnerStepsResult` (dataclass available in the eval environment):

| Field | Type | Description |
|-------|------|-------------|
| `final_logits` | `torch.Tensor` | Logits from the last forward pass. Must be 3D `(batch, seq_len-1, vocab)`. Cannot be `None`. |
| `total_tokens` | `int` | Total tokens processed across all steps (= `batch_size × seq_len × num_steps`). |
| `final_loss` | `float` | Loss value from the last training step. Must be positive and not NaN. |
| `final_state` | `dict` (required) | Full `state_dict` on CPU. **Required** for weight verification in all modes (single-GPU, DDP, FSDP, TP). |

### Multi-GPU Example (FSDP)

With the 7B benchmark model, miners **must** use a memory-sharding strategy. FSDP is recommended — it shards parameters, gradients, and optimizer states across GPUs.

When `num_gpus > 1`, `optimizer` is `None`. The miner wraps the model and creates its own optimizer:

```python
import functools
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def get_strategy():
    return {"dp_size": 2, "tp_size": 1}  # pure data parallel

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={type(model.model.layers[0])},
    )
    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                 auto_wrap_policy=wrap_policy, device_id=device.index)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    # ... training loop ...
    # Return gathered final_state for weight verification
```

See `local_test/train_fsdp.py`, `local_test/train_tp.py`, and `local_test/train_mixed.py` for complete working examples. The process group is already initialized by `torchrun` before `inner_steps` is called.

> **Note:** Single-GPU and DDP examples are kept in `local_test/` for reference but will OOM with the 7B model on A100 80 GB. Use FSDP or TP for the current benchmark.

### Rules

**General:**
- Process all tokens in each batch, return valid `final_logits` (not `None`)
- Train all model parameters (no freezing layers)
- Don't alias `torch` (e.g., `import torch as t`) -- the scanner needs the literal name
- `optimizer` is always `None` -- create your own after wrapping the model
- You may use any parallelism: DDP, FSDP, TP, PP, or combinations
- Must return `final_state` (full CPU `state_dict`) in `InnerStepsResult` for weight verification (required for all strategies)
- The validator verifies correctness via loss comparison and final weight verification against a reference run

A static security scanner blocks dangerous patterns (forbidden imports, monkey-patching, timer tampering, etc.). See [`src/crusades/core/security_defs.py`](src/crusades/core/security_defs.py) for the full blocklist. Run the Docker-based [`simulate_validator.py`](local_test/simulate_validator.py) before submitting to catch violations early.

Any genuine optimization is fair game -- `torch.compile`, mixed precision, Flash Attention, Triton kernels, CUDA Graphs, custom loss functions, and more. If the scanner rejects something that should be allowed, report to us.

> **Note:** The validator scans **all** code including `if __name__ == "__main__":` blocks. Use a separate test script if you need locally-useful imports (like `pathlib`) for testing.

---

## Configuration

Key settings in `hparams/hparams.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `netuid` | 3 | Subnet ID |
| `evaluation_runs` | 3 | Runs per submission (median taken) |
| `eval_steps` | 5 | Training steps per evaluation |
| `benchmark_model_name` | Qwen/Qwen2.5-3B | Model for evaluation |
| `benchmark_batch_size` | 16 | Batch size for evaluation |
| `docker.num_gpus` | 2 | Number of GPUs (multi-GPU via `torchrun`) |

---

## TUI Dashboard

Monitor crusades activity in real-time with the terminal dashboard.

```bash
# Connect to the official Crusades API
uv run -m crusades.tui --url 62.169.159.5:8080
```

### Features

- Leaderboard with MFU scores
- Recent submissions and their status
- MFU history chart
- Validator status
- View submission code (after evaluation)
