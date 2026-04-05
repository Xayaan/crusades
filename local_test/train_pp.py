# High-MFU Pipeline Parallelism strategy for 262K vocab
#
# Topology: dp_size=1, tp_size=1, pp_size=4 (requires 4 GPUs)
#   - Single 4-stage pipeline: rank 0 → rank 1 → rank 2 → rank 3
#   - All ranks process the SAME data (dp=1, no data-parallel all-reduce)
#
# Key optimisation: M=16 microbatching with standard 1F1B schedule.
#   Pipeline utilisation = M/(M+K-1) = 16/19 = 84.2%
#
# Under NCCL bandwidth throttling (P2P_DISABLE + MAX_NCHANNELS=1), PP
# dominates FSDP/TP/DDP because it only sends tiny activation tensors
# (~7 MB per micro) between adjacent stages.  There are ZERO collective
# operations (no all-reduce, no all-gather) — only point-to-point sends.
#
# Optimizations: torch.compile per stage, flash_attn CE (last stage),
# Selective Activation Checkpointing, bf16, pre-loaded batches,
# inductor/dynamo tuning, TF32 matmul, fused AdamW.

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

_NUM_MICROBATCHES = 16
_UNCHECKPOINT_LAST_N_PER_STAGE = 3


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None


def _sac_policy(ctx, func, *args, **kwargs):
    if func in {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def get_strategy():
    return {"dp_size": 1, "tp_size": 1, "pp_size": 4}


def _split_model_layers(model, pp_size):
    """Split layers evenly across pp_size stages."""
    n = len(model.model.layers)
    layers_per_stage = n // pp_size
    remainder = n % pp_size
    stages = []
    start = 0
    for s in range(pp_size):
        count = layers_per_stage + (1 if s < remainder else 0)
        stages.append(list(range(start, start + count)))
        start += count
    return stages


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = False
        if hasattr(model.config, "output_attentions"):
            model.config.output_attentions = False

    strategy = get_strategy()
    pp_size = strategy["pp_size"]

    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = device.index if device.index is not None else 0
    pp_rank = local_rank
    is_first_stage = pp_rank == 0
    is_last_stage = pp_rank == pp_size - 1
    pp_prev = pp_rank - 1 if not is_first_stage else -1
    pp_next = pp_rank + 1 if not is_last_stage else -1

    all_stage_layers = _split_model_layers(model, pp_size)
    my_layer_indices = all_stage_layers[pp_rank]

    all_layers = list(model.model.layers)

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

    def _run_layers(h):
        pos_emb = _compute_pos_emb(h)
        for i, fn in enumerate(layer_fns):
            if _sac_ctx is not None and i < num_ckpt:
                h = ckpt.checkpoint(fn, h, pos_emb, use_reentrant=False, context_fn=_sac_ctx)
            else:
                h = fn(h, pos_emb)
        return h

    if is_first_stage:
        _embed = model.model.embed_tokens

        def _stage_fwd(input_ids):
            h = _embed(input_ids)
            return _run_layers(h)

    elif is_last_stage:
        _norm = model.model.norm
        _head = model.lm_head

        def _stage_fwd(hidden):
            h = _run_layers(hidden)
            h = _norm(h)
            return _head(h)

    else:

        def _stage_fwd(hidden):
            return _run_layers(hidden)

    compiled_fwd = torch.compile(_stage_fwd, mode="default", dynamic=False)

    # --- No DP groups needed (dp=1).  PP group is default world group. ---

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
        )

    # --- Pre-load all batches and split into microbatches ---

    all_micro_inputs = []
    all_micro_labels = []
    tokens_per_batch = 0
    hidden_size = model.config.hidden_size
    n_microbatches = _NUM_MICROBATCHES

    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        inp = batch[:, :-1].contiguous()
        lab = batch[:, 1:].contiguous()
        tokens_per_batch = batch.numel()
        bs = inp.shape[0]
        micro_bs = max(1, bs // n_microbatches)
        step_inputs = []
        step_labels = []
        for m in range(0, bs, micro_bs):
            step_inputs.append(inp[m : m + micro_bs])
            step_labels.append(lab[m : m + micro_bs])
        all_micro_inputs.append(step_inputs)
        all_micro_labels.append(step_labels)

    torch.cuda.synchronize(device)
    total_tokens = num_steps * tokens_per_batch

    _ce = _flash_ce_inst
    final_logits = None
    final_loss = 0.0

    # 1F1B schedule parameters
    num_warmup = pp_size - 1 - pp_rank

    # --- Training loop with 1F1B schedule ---

    for step in range(num_steps):
        micro_inputs = all_micro_inputs[step]
        micro_labels = all_micro_labels[step]
        n_micro = len(micro_inputs)
        total_micro_tokens = sum(ml.numel() for ml in micro_labels)

        # saved_hidden[i]: data needed for backward of microbatch i
        #   first stage: the output tensor h (has grad_fn)
        #   middle stages: (input_hidden_with_grad, output_hidden_with_grad_fn)
        #   last stage: None (backward is immediate)
        saved = [None] * n_micro

        # isend handles for forward sends (non-blocking to avoid deadlock)
        fwd_reqs = [None] * n_micro

        step_loss = 0.0
        last_logits = None

        # ---- Helper: forward one microbatch ----
        def _do_forward(mb_idx):
            nonlocal step_loss, last_logits

            if is_first_stage:
                h = compiled_fwd(micro_inputs[mb_idx])
                saved[mb_idx] = h
                buf = h.detach().contiguous()
                fwd_reqs[mb_idx] = dist.isend(buf, dst=pp_next)

            elif is_last_stage:
                recv_buf = torch.zeros(
                    micro_inputs[mb_idx].shape[0],
                    micro_inputs[mb_idx].shape[1],
                    hidden_size,
                    device=device,
                    dtype=torch.bfloat16,
                )
                dist.recv(recv_buf, src=pp_prev)
                hidden_in = recv_buf.detach().requires_grad_(True)

                logits = compiled_fwd(hidden_in)
                micro_tokens = micro_labels[mb_idx].numel()
                loss = _ce(
                    logits.reshape(-1, logits.size(-1)),
                    micro_labels[mb_idx].reshape(-1),
                )
                weight = micro_tokens / total_micro_tokens
                scaled = loss * weight
                scaled.backward()

                dist.send(hidden_in.grad.contiguous(), dst=pp_prev)
                step_loss += scaled.detach().item()

                if mb_idx == n_micro - 1:
                    last_logits = logits.detach()

            else:
                recv_buf = torch.zeros(
                    micro_inputs[mb_idx].shape[0],
                    micro_inputs[mb_idx].shape[1],
                    hidden_size,
                    device=device,
                    dtype=torch.bfloat16,
                )
                dist.recv(recv_buf, src=pp_prev)
                hidden_in = recv_buf.detach().requires_grad_(True)

                h_out = compiled_fwd(hidden_in)
                saved[mb_idx] = (hidden_in, h_out)
                buf = h_out.detach().contiguous()
                fwd_reqs[mb_idx] = dist.isend(buf, dst=pp_next)

        # ---- Helper: backward one microbatch ----
        def _do_backward(mb_idx):
            if is_first_stage:
                if fwd_reqs[mb_idx] is not None:
                    fwd_reqs[mb_idx].wait()
                grad_buf = torch.zeros_like(saved[mb_idx])
                dist.recv(grad_buf, src=pp_next)
                saved[mb_idx].backward(grad_buf)
                saved[mb_idx] = None

            elif is_last_stage:
                pass  # backward already done in _do_forward

            else:
                if fwd_reqs[mb_idx] is not None:
                    fwd_reqs[mb_idx].wait()
                hidden_in, h_out = saved[mb_idx]
                grad_buf = torch.zeros_like(h_out)
                dist.recv(grad_buf, src=pp_next)
                h_out.backward(grad_buf)
                dist.send(hidden_in.grad.contiguous(), dst=pp_prev)
                saved[mb_idx] = None

        # ---- Phase 1: Warmup forwards ----
        for i in range(num_warmup):
            _do_forward(i)

        # ---- Phase 2: Steady state (1 backward + 1 forward) ----
        for i in range(num_warmup, n_micro):
            bwd_idx = i - num_warmup
            if not is_last_stage:
                _do_backward(bwd_idx)
            _do_forward(i)
            if is_last_stage:
                pass  # backward already happened inside _do_forward

        # ---- Phase 3: Cooldown backwards ----
        for i in range(n_micro - num_warmup, n_micro):
            _do_backward(i)

        if is_last_stage:
            final_loss = step_loss
            final_logits = last_logits

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # --- Send logits & loss from last stage → first stage ---

    if is_last_stage:
        if final_logits is None:
            final_logits = torch.zeros(1, device=device)
        loss_t = torch.tensor([final_loss], device=device, dtype=torch.float64)
        dist.send(loss_t, dst=0)
        dist.send(final_logits.contiguous(), dst=0)
    elif is_first_stage:
        vocab_size = model.config.vocab_size
        loss_t = torch.zeros(1, device=device, dtype=torch.float64)
        dist.recv(loss_t, src=pp_size - 1)
        final_loss = loss_t.item()
        micro_bs_last = all_micro_inputs[num_steps - 1][-1].shape[0]
        seq_last = all_micro_inputs[num_steps - 1][-1].shape[1]
        final_logits = torch.zeros(
            micro_bs_last, seq_last, vocab_size, device=device, dtype=torch.bfloat16
        )
        dist.recv(final_logits, src=pp_size - 1)

    # Middle stages need dummy logits/loss
    if not is_first_stage and not is_last_stage:
        final_logits = torch.zeros(1, device=device, dtype=torch.bfloat16)
        final_loss = 0.0

    full_state = _gather_pp_state(
        model,
        all_layers,
        my_layer_indices,
        pp_rank,
        pp_size,
        is_first_stage,
        is_last_stage,
        rank,
        device,
    )

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )


def _gather_pp_state(
    model,
    all_layers,
    my_layer_indices,
    pp_rank,
    pp_size,
    is_first_stage,
    is_last_stage,
    global_rank,
    device,
):
    """Gather full model state dict across all pipeline stages onto rank 0."""
    my_state = {}

    if is_first_stage:
        for k, v in model.model.embed_tokens.state_dict().items():
            my_state[f"model.embed_tokens.{k}"] = v.detach().clone()

    for idx in my_layer_indices:
        for k, v in all_layers[idx].state_dict().items():
            my_state[f"model.layers.{idx}.{k}"] = v.detach().clone()

    if is_last_stage:
        for k, v in model.model.norm.state_dict().items():
            my_state[f"model.norm.{k}"] = v.detach().clone()
        for k, v in model.lm_head.state_dict().items():
            my_state[f"lm_head.{k}"] = v.detach().clone()

    if not dist.is_initialized():
        return {k: v.cpu() for k, v in my_state.items()}

    my_keys = [(k, v.shape, v.dtype) for k, v in my_state.items()]

    # All ranks participate in each broadcast (collective requirement)
    all_keys = {}
    for src in range(pp_size):
        if src == global_rank:
            obj = [my_keys]
        else:
            obj = [None]
        dist.broadcast_object_list(obj, src=src)
        all_keys[src] = obj[0]

    if global_rank == 0:
        full_state = dict(my_state)
        for peer in range(1, pp_size):
            for k, shape, dtype in all_keys[peer]:
                buf = torch.empty(shape, dtype=dtype, device=device)
                dist.recv(buf, src=peer)
                full_state[k] = buf
        return {k: v.cpu() for k, v in full_state.items()}
    else:
        for k, v in my_state.items():
            dist.send(v.contiguous(), dst=0)
        return None
