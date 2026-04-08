import functools
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

try:
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
except Exception:
    pass


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None


def get_strategy():
    return {"dp_size": 4, "tp_size": 1, "ep_size": 1}


MICRO_BATCH_SIZE = 1


def _get_wrap_policy(model):
    layer_cls = set()
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        layer_cls.add(type(model.model.layers[0]))
    return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls)


def _prepare_model(model):
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = False
        if hasattr(model.config, "output_attentions"):
            model.config.output_attentions = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "preserve_rng_state": False,
            }
        )

    return model


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    _prepare_model(model)

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=_get_wrap_policy(model),
        mixed_precision=bf16_policy,
        device_id=device,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )

    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            weight_decay=0.1,
            momentum=0.0,
        )

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        micro_batches = [
            batch[i : i + MICRO_BATCH_SIZE] for i in range(0, batch.size(0), MICRO_BATCH_SIZE)
        ]
        num_accum = len(micro_batches)
        step_loss_sum = 0.0

        for idx, mb in enumerate(micro_batches):
            input_ids = mb[:, :-1].contiguous()
            labels = mb[:, 1:].contiguous()

            sync_ctx = model.no_sync() if idx < num_accum - 1 else nullcontext()
            with sync_ctx:
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                (loss / num_accum).backward()

            step_loss_sum += loss.item()
            if idx == num_accum - 1:
                final_logits = logits.detach()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_loss = step_loss_sum / num_accum

    rank = dist.get_rank() if dist.is_initialized() else 0
    full_state = None
    with FSDP.summon_full_params(model, writeback=False):
        raw = model.module if hasattr(model, "module") else model
        if rank == 0:
            sd = raw.state_dict()
            pinned = {k: torch.empty_like(v, device="cpu").pin_memory() for k, v in sd.items()}
            for k, v in sd.items():
                pinned[k].copy_(v, non_blocking=True)
            torch.cuda.synchronize(device)
            full_state = pinned

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )
