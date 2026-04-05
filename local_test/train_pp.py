# High-MFU Pipeline Parallelism strategy for 262K vocab
#
# Topology: dp_size=2, tp_size=1, pp_size=2 (requires 4 GPUs)
#   - 2 pipeline replicas, each with 2 stages
#   - Ranks [0,1] form pipeline 0; ranks [2,3] form pipeline 1
#   - Different pipelines get different data (data-parallel across DP dim)
#
# Manual 1F1B schedule: each stage holds half the transformer layers.
# Stage 0: embed + first-half layers  →  send activations
# Stage 1: recv activations  →  second-half layers + norm + lm_head  →  loss
# Backward: reverse with gradient communication.
#
# Under NCCL bandwidth throttling (NCCL_P2P_DISABLE=1), PP outperforms
# FSDP/TP because it only sends activation tensors (~112 MB/step) instead
# of all-reducing all gradients (~14 GB/step).
#
# Optimizations: torch.compile per stage, flash_attn CE (last stage),
# Selective Activation Checkpointing, bf16, pre-loaded batches,
# inductor/dynamo tuning, TF32 matmul, fused AdamW, lm_head graph break.

import functools
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.utils.checkpoint as ckpt

try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

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


_UNCHECKPOINT_LAST_N_PER_STAGE = 4


def _sac_policy(ctx, func, *args, **kwargs):
    if func in {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def get_strategy():
    return {"dp_size": 2, "tp_size": 1, "pp_size": 2}


def _split_model_layers(model):
    n = len(model.model.layers)
    mid = n // 2
    return list(range(mid)), list(range(mid, n))


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = False
        if hasattr(model.config, "output_attentions"):
            model.config.output_attentions = False

    strategy = get_strategy()
    pp_size = strategy["pp_size"]
    dp_size = strategy["dp_size"]

    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = device.index if device.index is not None else 0
    pp_rank = local_rank % pp_size
    is_first_stage = pp_rank == 0
    is_last_stage = pp_rank == pp_size - 1
    pp_peer = local_rank + 1 if is_first_stage else local_rank - 1

    layer_indices_0, layer_indices_1 = _split_model_layers(model)
    my_layer_indices = layer_indices_0 if is_first_stage else layer_indices_1

    all_layers = list(model.model.layers)
    n_layers = len(all_layers)

    model = model.to(device=device, dtype=torch.bfloat16)

    for i, layer in enumerate(all_layers):
        if i not in my_layer_indices:
            for p in layer.parameters():
                p.requires_grad_(False)
                p.data = torch.empty(0, dtype=p.dtype, device=device)

    for idx in my_layer_indices:
        layer = all_layers[idx]
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = 0

    # --- Build compiled stage forward with SAC ---

    _rotary_emb = model.model.rotary_emb if hasattr(model.model, "rotary_emb") else None

    def _make_layer_fn(layer):
        def fn(h, pos_emb):
            out = layer(h, position_embeddings=pos_emb)
            return out[0] if isinstance(out, tuple) else out

        return fn

    layer_fns = [_make_layer_fn(all_layers[i]) for i in my_layer_indices]

    num_owned = len(my_layer_indices)
    num_ckpt = max(0, num_owned - _UNCHECKPOINT_LAST_N_PER_STAGE)
    _sac_ctx = (
        functools.partial(create_selective_checkpoint_contexts, _sac_policy) if _HAS_SAC else None
    )

    def _compute_pos_emb(h):
        if _rotary_emb is not None:
            pos_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
            return _rotary_emb(h, pos_ids)
        return None

    if is_first_stage:
        _embed = model.model.embed_tokens

        def _stage_fwd(input_ids):
            h = _embed(input_ids)
            pos_emb = _compute_pos_emb(h)
            for i, fn in enumerate(layer_fns):
                if _sac_ctx is not None and i < num_ckpt:
                    h = ckpt.checkpoint(fn, h, pos_emb, use_reentrant=False, context_fn=_sac_ctx)
                else:
                    h = fn(h, pos_emb)
            return h

    else:
        _norm = model.model.norm
        _head = model.lm_head

        @torch._dynamo.disable(recursive=False)
        def _eager_lm_head(h):
            return _head(h)

        def _stage_fwd(hidden):
            h = hidden
            pos_emb = _compute_pos_emb(h)
            for i, fn in enumerate(layer_fns):
                if _sac_ctx is not None and i < num_ckpt:
                    h = ckpt.checkpoint(fn, h, pos_emb, use_reentrant=False, context_fn=_sac_ctx)
                else:
                    h = fn(h, pos_emb)
            h = _norm(h)
            return _eager_lm_head(h)

    compiled_fwd = torch.compile(_stage_fwd, mode="default", dynamic=False)

    # --- DP group, optimizer, pre-load ---

    dp_group = None
    if dp_size > 1:
        dp_ranks = [r for r in range(num_gpus) if (r % pp_size) == pp_rank]
        dp_group = dist.new_group(dp_ranks)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            trainable_params,
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

    hidden_size = model.config.hidden_size
    _ce = _flash_ce_inst
    final_logits = None
    final_loss = 0.0

    # --- Training loop ---

    for step in range(num_steps):
        bs, seq_len = all_inputs[step].shape

        if is_first_stage:
            hidden = compiled_fwd(all_inputs[step])
            dist.send(hidden.detach().contiguous(), dst=pp_peer)

            recv_grad = torch.zeros_like(hidden)
            dist.recv(recv_grad, src=pp_peer)
            hidden.backward(recv_grad)

        else:
            recv_buf = torch.zeros(bs, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
            dist.recv(recv_buf, src=pp_peer)
            hidden = recv_buf.detach().requires_grad_(True)

            logits = compiled_fwd(hidden)
            loss = _ce(logits.reshape(-1, logits.size(-1)), all_labels[step].reshape(-1))
            loss.backward()

            dist.send(hidden.grad.contiguous(), dst=pp_peer)
            final_logits = logits.detach()
            final_loss = loss.item()

        if dp_group is not None:
            for p in trainable_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=dp_group)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # --- Send logits & loss from last stage → first stage ---

    if is_last_stage:
        if final_logits is None:
            final_logits = torch.zeros(1, device=device)
        loss_t = torch.tensor([final_loss], device=device, dtype=torch.float64)
        dist.send(loss_t, dst=pp_peer)
        dist.send(final_logits.contiguous(), dst=pp_peer)
    elif is_first_stage:
        vocab_size = model.config.vocab_size
        loss_t = torch.zeros(1, device=device, dtype=torch.float64)
        dist.recv(loss_t, src=pp_peer)
        final_loss = loss_t.item()
        final_logits = torch.zeros(bs, seq_len, vocab_size, device=device, dtype=torch.bfloat16)
        dist.recv(final_logits, src=pp_peer)

    full_state = _gather_pp_state(
        model, all_layers, my_layer_indices, n_layers, pp_rank, pp_peer, is_first_stage, rank
    )

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )


def _gather_pp_state(
    model, all_layers, my_layer_indices, n_layers, pp_rank, pp_peer, is_first_stage, global_rank
):
    """Gather full model state dict across pipeline stages onto rank 0."""
    my_state = {}

    if is_first_stage:
        for k, v in model.model.embed_tokens.state_dict().items():
            my_state[f"model.embed_tokens.{k}"] = v.detach().cpu().clone()

    for idx in my_layer_indices:
        for k, v in all_layers[idx].state_dict().items():
            my_state[f"model.layers.{idx}.{k}"] = v.detach().cpu().clone()

    if not is_first_stage:
        for k, v in model.model.norm.state_dict().items():
            my_state[f"model.norm.{k}"] = v.detach().cpu().clone()
        for k, v in model.lm_head.state_dict().items():
            my_state[f"lm_head.{k}"] = v.detach().cpu().clone()

    if not dist.is_initialized():
        return my_state

    if is_first_stage:
        keys_and_shapes = [(k, v.shape, v.dtype) for k, v in my_state.items()]
        obj_list = [keys_and_shapes]
        dist.broadcast_object_list(obj_list, src=dist.get_rank())

        peer_obj = [None]
        dist.broadcast_object_list(peer_obj, src=pp_peer)
        peer_keys = peer_obj[0]

        for k, shape, dtype in peer_keys:
            buf = torch.empty(shape, dtype=dtype)
            dist.recv(buf, src=pp_peer)
            my_state[k] = buf

        return my_state if global_rank == 0 else None
    else:
        peer_obj = [None]
        dist.broadcast_object_list(peer_obj, src=pp_peer)

        keys_and_shapes = [(k, v.shape, v.dtype) for k, v in my_state.items()]
        obj_list = [keys_and_shapes]
        dist.broadcast_object_list(obj_list, src=dist.get_rank())

        for k, v in my_state.items():
            dist.send(v.contiguous(), dst=pp_peer)

        return None
