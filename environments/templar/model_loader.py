"""Model loading and caching for evaluation environment."""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

# Reference to the shared evaluation cache (injected by env.py via set_cache)
_CACHE: dict = {}


def set_cache(cache: dict) -> None:
    """Point this module's _CACHE at the shared env-level cache dict."""
    global _CACHE
    _CACHE = cache


def _load_model(model_path: str, use_random_init: bool = False):
    """Load model from HuggingFace.

    Args:
        model_path: HuggingFace model name/path
        use_random_init: If True, initialize with random weights

    Note:
        Docker evaluation runs with --network=none. Models must be
        pre-cached in the Docker image. If cache is missing, rebuild:
        docker build --network=host -f environments/templar/Dockerfile -t templar-eval:latest .
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if use_random_init:
        # Random initialization
        logger.info(f"Loading model config from {model_path} with RANDOM initialization")

        # Use local cache only (Docker runs with --network=none)
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        device = f"cuda:{_LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        model = model.to(device)
        logger.info(f"Model loaded on {device} (rank {_LOCAL_RANK})")
    else:
        logger.info(f"Loading pretrained model from {model_path}")
        # Use local cache only (Docker runs with --network=none)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )

    model.train()

    # Ensure ALL parameters have requires_grad=True for training
    for param in model.parameters():
        param.requires_grad = True

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def _count_model_params(model: torch.nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def _get_cached_model(model_path: str, use_random_init: bool = False):
    """Get model from cache or load it."""
    cached = _CACHE.get("model")
    cached_path = _CACHE.get("model_path")
    cached_random_init = _CACHE.get("use_random_init")

    # Cache hit only if path AND init mode match
    if cached is not None and cached_path == model_path and cached_random_init == use_random_init:
        if _CACHE.get("initial_state"):
            cached.load_state_dict(_CACHE["initial_state"])
            cached.to(torch.device(f"cuda:{_LOCAL_RANK}"))
        return cached

    model = _load_model(model_path, use_random_init=use_random_init)
    _CACHE["model"] = model
    _CACHE["model_path"] = model_path
    _CACHE["use_random_init"] = use_random_init
    _CACHE["initial_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model
