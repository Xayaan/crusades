# Reference: Mixed DP+TP (Data Parallel + Tensor Parallel) strategy
#
# Topology: dp_size=2, tp_size=2 (requires 4 GPUs)
#   - 2D mesh: ranks [0,1] form TP group 0, ranks [2,3] form TP group 1
#   - Each TP group gets different data (data-parallel across DP dim)
#   - Within each TP group, tensors are sharded (tensor-parallel)
#   - Equivalent to: get_strategy() -> {"dp_size": 2, "tp_size": 2}
#
# Neither FSDP nor DDP can wrap TP's DTensor parameters (both try to
# flatten/view params, which DTensor's sharding propagation rejects).
# Instead we manually all-reduce gradients across the DP process group
# after each backward pass.
#
# Requirements for verification:
#   - get_strategy() returning {"dp_size": 2, "tp_size": 2}
#   - Return InnerStepsResult with final_logits, total_tokens, final_loss
#   - Must return final_state: gathered full tensors from DTensor shards

from dataclasses import dataclass

import torch
import torch.distributed as dist
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
    return {"dp_size": 2, "tp_size": 2}


def _apply_tp(model, tp_mesh):
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "o_proj"):
            parallelize_module(
                module,
                tp_mesh,
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
                tp_mesh,
                {
                    "gate_proj": ColwiseParallel(),
                    "up_proj": ColwiseParallel(),
                    "down_proj": RowwiseParallel(),
                },
            )
    return model


def _allreduce_grads(model, dp_pg):
    """Average gradients across DP group (replaces DDP/FSDP gradient sync).

    TP converts parameters to DTensors; calling dist.all_reduce on a DTensor
    triggers DTensor dispatch which fails without a matching DeviceMesh.
    We operate on the underlying local tensor shard instead.
    """
    for param in model.parameters():
        if param.grad is not None:
            g = param.grad
            local_g = g._local_tensor if hasattr(g, "_local_tensor") else g
            dist.all_reduce(local_g, op=dist.ReduceOp.AVG, group=dp_pg)


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

    strategy = get_strategy()
    expected_gpus = strategy["dp_size"] * strategy["tp_size"]
    if num_gpus != expected_gpus:
        raise ValueError(
            f"get_strategy() requires {expected_gpus} GPUs "
            f"(dp_size={strategy['dp_size']} * tp_size={strategy['tp_size']}), "
            f"but num_gpus={num_gpus}"
        )

    is_multi = num_gpus > 1
    dp_pg = None
    dp_size = 1

    if is_multi:
        dp_size = strategy["dp_size"]
        tp_size = strategy["tp_size"]

        mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        tp_mesh = mesh_2d["tp"]
        dp_pg = mesh_2d.get_group("dp")

        model = _apply_tp(model, tp_mesh)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=False,
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

        if dp_pg is not None:
            _allreduce_grads(model, dp_pg)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_logits = logits.detach()
        final_loss = loss.item()

    if is_multi:
        rank = dist.get_rank() if dist.is_initialized() else 0
        gathered = _gather_full_state(model)
        full_state = gathered if rank == 0 else None
    else:
        full_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )
