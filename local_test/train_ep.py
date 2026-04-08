from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None


def get_strategy():
    return {"dp_size": 1, "tp_size": 1, "ep_size": 4}


class _EPAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class EPMoeBlock(torch.nn.Module):
    def __init__(self, original_moe, ep_rank, ep_size, ep_group):
        super().__init__()
        self.gate = original_moe.gate
        self.num_experts = original_moe.num_experts
        self.top_k = original_moe.top_k
        self.norm_topk_prob = original_moe.norm_topk_prob
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.experts_per_rank = self.num_experts // ep_size
        self.local_start = ep_rank * self.experts_per_rank

        self.experts = torch.nn.ModuleList(
            [original_moe.experts[self.local_start + i] for i in range(self.experts_per_rank)]
        )

    def forward(self, hidden_states):
        bsz, seq, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)

        router_logits = self.gate(hidden_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_flat.dtype)

        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 1, 0)

        output = torch.zeros_like(hidden_flat)
        for e_local in range(self.experts_per_rank):
            e_global = self.local_start + e_local
            idx, top_x = torch.where(expert_mask[e_global])
            if top_x.numel() == 0:
                continue
            current_out = self.experts[e_local](hidden_flat[top_x])
            current_out = current_out * topk_weights[top_x, idx].unsqueeze(-1)
            output.index_add_(0, top_x, current_out.to(output.dtype))

        output = _EPAllReduce.apply(output, self.ep_group)
        return output.reshape(bsz, seq, hidden_dim), router_logits


def _shard_experts(model, ep_rank, ep_size, ep_group):
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "gate") and hasattr(mlp, "experts") and hasattr(mlp, "num_experts"):
            assert mlp.num_experts % ep_size == 0
            layer.mlp = EPMoeBlock(mlp, ep_rank, ep_size, ep_group)

    torch.cuda.empty_cache()


def _sync_replicated_grads(model, ep_group):
    world = dist.get_world_size(ep_group)
    for name, param in model.named_parameters():
        if param.grad is not None and "experts" not in name:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=ep_group)
            param.grad.div_(world)


def _gather_full_state_dict(model, ep_rank, ep_size, device):
    config = model.config
    experts_per_rank = config.num_experts // ep_size

    expert_items = sorted(
        [(n, p.data) for n, p in model.named_parameters() if "experts" in n],
        key=lambda x: x[0],
    )

    non_expert_state = {}
    for name, param in model.named_parameters():
        if "experts" not in name:
            non_expert_state[name] = param.data.detach().cpu().clone()
    for name, buf in model.named_buffers():
        if "experts" not in name:
            non_expert_state[name] = buf.data.detach().cpu().clone()

    shapes = [t.shape for _, t in expert_items]
    local_flat = torch.cat([t.reshape(-1) for _, t in expert_items]).contiguous()

    full_sd = dict(non_expert_state) if ep_rank == 0 else None

    for r in range(ep_size):
        buf = local_flat.clone() if ep_rank == r else torch.empty_like(local_flat)
        dist.broadcast(buf, src=r)

        if ep_rank == 0:
            offset = 0
            for (name, _), shape in zip(expert_items, shapes):
                numel = shape.numel()
                tensor = buf[offset : offset + numel].reshape(shape)

                parts = name.split(".")
                idx_pos = parts.index("experts") + 1
                local_idx = int(parts[idx_pos])
                global_idx = r * experts_per_rank + local_idx
                parts[idx_pos] = str(global_idx)
                full_sd[".".join(parts)] = tensor.detach().cpu().clone()

                offset += numel

    return full_sd


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

    ep_size = num_gpus
    ep_rank = dist.get_rank() if dist.is_initialized() else 0
    ep_group = dist.group.WORLD

    _shard_experts(model, ep_rank, ep_size, ep_group)

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
        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()

        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()

        _sync_replicated_grads(model, ep_group)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_logits = logits.detach()
        final_loss = loss.item()

    full_state = _gather_full_state_dict(model, ep_rank, ep_size, device)

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )
