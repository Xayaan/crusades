from dataclasses import dataclass

import torch
import torch.distributed as dist
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
    return {"dp_size": 2, "tp_size": 2, "ep_size": 1}


def _apply_tp(model, tp_mesh):
    for _name, module in model.named_modules():
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
    for param in model.parameters():
        if param.grad is not None:
            g = param.grad
            local_g = g._local_tensor if hasattr(g, "_local_tensor") else g
            dist.all_reduce(local_g, op=dist.ReduceOp.AVG, group=dp_pg)


def _gather_full_state(model):
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


def _chunked_lm_loss(lm_head, hidden_states, labels, chunk_tokens=2048):
    _b, _s, h = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, h)
    labels_flat = labels.reshape(-1)
    total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
    n_valid = torch.tensor(0, device=hidden_states.device, dtype=torch.long)

    for i in range(0, hidden_flat.size(0), chunk_tokens):
        chunk_h = hidden_flat[i : i + chunk_tokens]
        chunk_labels = labels_flat[i : i + chunk_tokens]
        chunk_logits = lm_head(chunk_h)
        mask = chunk_labels != -100
        if mask.any():
            total_loss = total_loss + torch.nn.functional.cross_entropy(
                chunk_logits[mask], chunk_labels[mask], reduction="sum"
            )
            n_valid = n_valid + mask.sum()
        del chunk_logits
    return total_loss / n_valid.clamp(min=1)


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "preserve_rng_state": False,
            }
        )

    model = model.to(dtype=torch.bfloat16)

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

    if is_multi:
        dp_size = strategy["dp_size"]
        tp_size = strategy["tp_size"]

        mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        tp_mesh = mesh_2d["tp"]
        dp_pg = mesh_2d.get_group("dp")

        model = _apply_tp(model, tp_mesh)

    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            weight_decay=0.1,
            momentum=0.0,
        )

    lm_head = model.lm_head

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        hidden_states = model.model(input_ids)[0]
        loss = _chunked_lm_loss(lm_head, hidden_states, labels)

        loss.backward()

        if dp_pg is not None:
            _allreduce_grads(model, dp_pg)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        with torch.no_grad():
            final_logits = lm_head(hidden_states).detach()
        final_loss = loss.item()
        del hidden_states

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
