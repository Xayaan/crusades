# Reference: TP (Tensor Parallel) strategy
#
# Topology: dp_size=1, tp_size=num_gpus
#   - All ranks receive the same data (NOT data-parallel)
#   - Equivalent to: get_strategy() -> {"dp_size": 1, "tp_size": num_gpus}
#
# Requirements for verification:
#   - get_strategy() returning "tp" or {"dp_size": 1, "tp_size": N}
#   - Return InnerStepsResult with final_logits, total_tokens, final_loss
#   - Must return final_state: gathered full tensors from DTensor shards
#     TP replaces params with DTensors so validator cannot read weights directly

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None


def get_strategy():
    # Pure tensor-parallel: all GPUs get the same data.
    # Use "tp" for any GPU count, or a dict for a fixed topology.
    return {"dp_size": 1, "tp_size": 4}


def _apply_tp(model, device_mesh):
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "o_proj"):
            parallelize_module(
                module,
                device_mesh,
                {
                    "q_proj": ColwiseParallel(),
                    "k_proj": ColwiseParallel(),
                    "v_proj": ColwiseParallel(),
                    "o_proj": RowwiseParallel(),
                },
            )
        if hasattr(module, "gate_proj") and hasattr(module, "down_proj"):
            parallelize_module(
                module,
                device_mesh,
                {
                    "gate_proj": ColwiseParallel(),
                    "up_proj": ColwiseParallel(),
                    "down_proj": RowwiseParallel(),
                },
            )
    return model


def _gather_full_state(model):
    """Gather full tensors from DTensor shards (collective op, all ranks must call)."""
    state = {}
    for name, param in model.named_parameters():
        p = param.data
        if hasattr(p, "full_tensor"):
            p = p.full_tensor()
        state[name] = p.detach().cpu().clone()
    for name, buf in model.named_buffers():
        b = buf.data
        if hasattr(b, "full_tensor"):
            b = b.full_tensor()
        state[name] = b.detach().cpu().clone()
    sd = model.state_dict()
    for key in sd:
        if key not in state:
            val = sd[key]
            if hasattr(val, "full_tensor"):
                val = val.full_tensor()
            state[key] = val.detach().cpu().clone()
    return state


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if hasattr(model, "config"):
        model.config.use_cache = False

    is_tp = num_gpus > 1
    if is_tp:
        mesh = init_device_mesh("cuda", (num_gpus,))
        model = _apply_tp(model, mesh)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=not is_tp,
        )

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_logits = logits.detach()
        final_loss = loss.item()

    # Gather full state dict for weight verification.
    # TP requires full_tensor() calls (collective); single-GPU can read directly.
    if is_tp:
        import torch.distributed as dist

        gathered = _gather_full_state(model)
        rank = dist.get_rank() if dist.is_initialized() else 0
        full_state = gathered if rank == 0 else None
    else:
        full_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )
