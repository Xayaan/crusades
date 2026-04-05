# High-MFU Mixed DP+TP strategy for 262K vocab
#
# Topology: dp_size=2, tp_size=2, pp_size=1 (requires 4 GPUs)
#   - 2D mesh: ranks [0,1] form TP group 0, ranks [2,3] form TP group 1
#   - Each TP group gets different data (data-parallel across DP dim)
#   - Within each TP group, tensors are sharded (tensor-parallel)
#
# Neither FSDP nor DDP can wrap TP's DTensor parameters.
# Gradients are manually all-reduced across the DP process group.
#
# Optimizations: Selective Activation Checkpointing, bf16, pre-loaded
# batches, flash_attn CE, TF32 matmul.

import functools
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.utils.checkpoint as ckpt
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

try:
    from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy

    _HAS_SAC = True
except ImportError:
    _HAS_SAC = False


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

try:
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
except Exception:
    pass

from flash_attn.losses.cross_entropy import CrossEntropyLoss as _FlashCELoss

_flash_ce_inst = _FlashCELoss(ignore_index=-100)


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None


_PREPARED = set()
_UNCHECKPOINT_LAST_N = 8


def _sac_policy(ctx, func, *args, **kwargs):
    if func in {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


class _AllSAC:
    def __init__(self, num_ckpt_layers):
        self.num_ckpt_layers = num_ckpt_layers
        self._count = 0

    def __call__(self, fn, *args, **kwargs):
        self._count += 1
        ctx_fn = functools.partial(create_selective_checkpoint_contexts, _sac_policy)
        return ckpt.checkpoint(fn, *args, use_reentrant=False, context_fn=ctx_fn, **kwargs)


def get_strategy():
    return {"dp_size": 2, "tp_size": 2, "pp_size": 1}


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
    """Average gradients across DP group using underlying local tensor shard."""
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


def _prepare_model(model):
    mid = id(model)
    if mid in _PREPARED:
        return
    _PREPARED.add(mid)
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = False
        if hasattr(model.config, "output_attentions"):
            model.config.output_attentions = False

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
        num_ckpt_layers = num_layers - _UNCHECKPOINT_LAST_N

        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = 0

        if _HAS_SAC and num_ckpt_layers > 0:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": False,
                        "preserve_rng_state": False,
                    }
                )
            for idx, layer in enumerate(model.model.layers):
                if hasattr(layer, "gradient_checkpointing") and idx >= num_ckpt_layers:
                    layer.gradient_checkpointing = False
            model.model._gradient_checkpointing_func = _AllSAC(num_ckpt_layers)


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    _prepare_model(model)
    model = model.to(dtype=torch.bfloat16)

    strategy = get_strategy()
    expected_gpus = strategy["dp_size"] * strategy["tp_size"] * strategy.get("pp_size", 1)
    if num_gpus != expected_gpus:
        raise ValueError(
            f"get_strategy() requires {expected_gpus} GPUs "
            f"(dp={strategy['dp_size']}*tp={strategy['tp_size']}*pp={strategy.get('pp_size', 1)}), "
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
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=False,
        )

    _backbone = model.model
    _head = model.lm_head
    _ce = _flash_ce_inst

    all_inputs = []
    all_labels = []
    tokens_per_batch = 0
    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        all_inputs.append(batch[:, :-1].contiguous())
        all_labels.append(batch[:, 1:].contiguous())
        tokens_per_batch = batch.numel()

    torch.cuda.synchronize(device)
    total_tokens = num_steps * tokens_per_batch

    for step in range(num_steps):
        hidden = _backbone(all_inputs[step])[0]
        logits = _head(hidden)
        loss = _ce(logits.reshape(-1, logits.size(-1)), all_labels[step].reshape(-1))

        loss.backward()

        if dp_pg is not None:
            _allreduce_grads(model, dp_pg)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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
