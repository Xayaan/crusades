# High-MFU DDP strategy for 262K vocab (google/gemma-3-27b-it tokenizer)
#
# Topology: dp_size=4, tp_size=1, pp_size=1 (DDP across 4 GPUs)
#
# DDP replicates the full model per GPU. With 262K vocab the resized model
# is ~8.4B params, so memory is tight: micro-batching with gradient
# accumulation keeps peak VRAM under 80 GB.
#
# Optimizations: torch.compile, flash_attn CE, Selective Activation
# Checkpointing, bf16 logits (no fp32 upcast), pre-loaded batches,
# inductor/dynamo tuning, TF32 matmul, fused AdamW.

import functools
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.utils.checkpoint as ckpt
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817

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

try:
    import torch._inductor.config as _ind_cfg

    _ind_cfg.coordinate_descent_tuning = True
    _ind_cfg.triton.unique_kernel_names = True
    _ind_cfg.fx_graph_cache = True
    _ind_cfg.triton.cudagraph_trees = True
    _ind_cfg.epilogue_fusion = True
    _ind_cfg.shape_padding = True
except Exception:
    pass

try:
    import torch._dynamo.config as _dyn_cfg

    _dyn_cfg.cache_size_limit = 128
    _dyn_cfg.suppress_errors = True
    _dyn_cfg.assume_static_by_default = True
    _dyn_cfg.automatic_dynamic_shapes = False
    _dyn_cfg.optimize_ddp = True
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
MICRO_BATCH_SIZE = 1


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
    return {"dp_size": 4, "tp_size": 1, "pp_size": 1}


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

    if hasattr(model, "lm_head") and hasattr(model, "model"):
        _backbone = model.model
        _head = model.lm_head

        @torch._dynamo.disable(recursive=False)
        def _eager_lm_head(hidden):
            return _head(hidden)

        def _bf16_forward(input_ids, **kwargs):
            return _eager_lm_head(_backbone(input_ids)[0])

        model.forward = _bf16_forward


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    _prepare_model(model)
    model = model.to(dtype=torch.bfloat16)

    if num_gpus > 1:
        model = DDP(model, device_ids=[device.index])

    def fwd_fn(input_ids):
        return model(input_ids)

    compiled_fwd = torch.compile(fwd_fn, mode="default", dynamic=False)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
        )

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
    _ce = _flash_ce_inst

    for step in range(num_steps):
        inp = all_inputs[step]
        lab = all_labels[step]
        micro_batches_in = [
            inp[i : i + MICRO_BATCH_SIZE] for i in range(0, inp.size(0), MICRO_BATCH_SIZE)
        ]
        micro_batches_lab = [
            lab[i : i + MICRO_BATCH_SIZE] for i in range(0, lab.size(0), MICRO_BATCH_SIZE)
        ]
        num_accum = len(micro_batches_in)

        for i in range(num_accum):
            no_sync = hasattr(model, "no_sync") and i < num_accum - 1
            ctx = model.no_sync() if no_sync else nullcontext()

            with ctx:
                logits = compiled_fwd(micro_batches_in[i])
                loss = _ce(logits.reshape(-1, logits.size(-1)), micro_batches_lab[i].reshape(-1))
                (loss / num_accum).backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    final_logits = logits.detach()
    final_loss = loss.item()

    raw_model = model.module if hasattr(model, "module") else model
    full_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )
