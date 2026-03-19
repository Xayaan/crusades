# Reference: DDP (Distributed Data Parallel) strategy
#
# Uses micro-batch gradient accumulation for Qwen2.5-3B on A100-80GB.
# DDP replicates the full model per GPU — memory is tight (~77GB peak).
# micro_batch=1 keeps activations small enough to leave headroom.
#
# Topology: dp_size=num_gpus, tp_size=1
#   - Each rank processes different data (data-parallel)
#   - Equivalent to: get_strategy() -> {"dp_size": num_gpus, "tp_size": 1}
#
# Requirements for verification:
#   - get_strategy() returning "ddp" or {"dp_size": N, "tp_size": 1}
#   - Return InnerStepsResult with final_logits, total_tokens, final_loss
#   - Must return final_state with full model state_dict for weight verification

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None


def get_strategy():
    # Pure data-parallel: every GPU gets different data.
    # Use "ddp" for any GPU count, or a dict for a fixed topology.
    return {"dp_size": 2, "tp_size": 1}


MICRO_BATCH_SIZE = 1


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if num_gpus > 1:
        model = DDP(model, device_ids=[device.index])

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
        )

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long)
        micro_batches = [
            batch[i : i + MICRO_BATCH_SIZE] for i in range(0, batch.size(0), MICRO_BATCH_SIZE)
        ]
        num_accum = len(micro_batches)
        step_loss_sum = 0.0

        for i, mb in enumerate(micro_batches):
            input_ids = mb[:, :-1]
            labels = mb[:, 1:]

            no_sync = hasattr(model, "no_sync") and i < num_accum - 1
            ctx = model.no_sync() if no_sync else nullcontext()

            with ctx:
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                (loss / num_accum).backward()

            step_loss_sum += loss.item()
            if i == num_accum - 1:
                final_logits = logits.detach()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_loss = step_loss_sum / num_accum

    raw_model = model.module if hasattr(model, "module") else model
    full_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )
