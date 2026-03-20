# Reference: FSDP (Fully Sharded Data Parallel) strategy
#
# Topology: dp_size=num_gpus, tp_size=1
#   - Each rank processes different data (data-parallel)
#   - Equivalent to: get_strategy() -> {"dp_size": num_gpus, "tp_size": 1}
#
# Requirements for verification:
#   - get_strategy() returning "fsdp" or {"dp_size": N, "tp_size": 1}
#   - Return InnerStepsResult with final_logits, total_tokens, final_loss
#   - Must return final_state: gathered full state dict for weight verification
#     FSDP flattens params so validator cannot read weights directly

import functools
import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None


def get_strategy():
    # Pure data-parallel (sharded): every GPU gets different data.
    # Use "fsdp" for any GPU count, or a dict for a fixed topology.
    return {"dp_size": 4, "tp_size": 1}


def _get_wrap_policy(model):
    layer_cls = None
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        layer_cls = model.model.layers[0].__class__
    elif (
        hasattr(model, "transformer")
        and hasattr(model.transformer, "h")
        and len(model.transformer.h) > 0
    ):
        layer_cls = model.transformer.h[0].__class__

    if layer_cls is None:
        return None

    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_cls},
    )


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if hasattr(model, "config"):
        model.config.use_cache = False

    is_fsdp = num_gpus > 1
    if is_fsdp:
        wrap_policy = _get_wrap_policy(model)
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=mp_policy,
            device_id=device,
            use_orig_params=True,
        )

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=not is_fsdp,
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
    # FSDP requires the FULL_STATE_DICT context; single-GPU can read directly.
    if is_fsdp:
        rank = dist.get_rank() if dist.is_initialized() else 0
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                sd = model.state_dict()
                full_state = sd if rank == 0 else None
    else:
        full_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )
