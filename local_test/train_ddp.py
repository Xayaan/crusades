from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None


def get_strategy():
    return {"dp_size": 4, "tp_size": 1, "ep_size": 1}


MICRO_BATCH_SIZE = 1


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

    if num_gpus > 1:
        model = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True)

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
        batch = next(data_iterator).to(device, dtype=torch.long)
        micro_batches = [
            batch[i : i + MICRO_BATCH_SIZE] for i in range(0, batch.size(0), MICRO_BATCH_SIZE)
        ]
        num_accum = len(micro_batches)
        step_loss_sum = 0.0

        for i, mb in enumerate(micro_batches):
            input_ids = mb[:, :-1]
            labels = mb[:, 1:]

            no_sync = hasattr(model, "no_sync") and i < num_accum - 1
            ctx = model.no_sync() if no_sync else nullcontext()

            with ctx:
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                (loss / num_accum).backward()

            step_loss_sum += loss.item()
            if i == num_accum - 1:
                final_logits = logits.detach()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_loss = step_loss_sum / num_accum

    raw_model = model.module if hasattr(model, "module") else model
    full_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )
