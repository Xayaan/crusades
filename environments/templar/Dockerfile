# Usage: docker build --network=host -f environments/templar/Dockerfile -t templar-eval:latest .
# Run from repo root so hparams.json is accessible

FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git curl ca-certificates build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash appuser
WORKDIR /app

COPY --chown=appuser:appuser environments/templar/requirements.txt /app/
USER appuser
RUN pip install --no-cache-dir --user -r requirements.txt

# flash-attn pre-built wheel (torch 2.9 + CUDA 12.6 + Python 3.11)
RUN pip install --no-cache-dir --user \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu126torch2.9-cp311-cp311-linux_x86_64.whl"

USER root
RUN apt-get purge -y g++ make dpkg-dev && apt-get autoremove -y \
    && apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*
USER appuser

COPY --chown=appuser:appuser hparams/hparams.json /app/

RUN python -c "\
import json; \
cfg = json.load(open('/app/hparams.json')); \
model = cfg['benchmark_model_name']; \
from transformers import AutoModelForCausalLM, AutoTokenizer; \
print(f'Downloading model: {model}'); \
AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto', trust_remote_code=True); \
AutoTokenizer.from_pretrained(model, trust_remote_code=True); \
print('Model cached')"

COPY --chown=appuser:appuser environments/templar/cache_dataset.py /app/
RUN python -c "\
import json, os; \
cfg = json.load(open('/app/hparams.json')); \
os.environ['DATASET_NAME'] = cfg['benchmark_dataset_name']; \
os.environ['NUM_SAMPLES'] = str(cfg['benchmark_data_samples']); \
exec(open('/app/cache_dataset.py').read())" || [ -f /home/appuser/.cache/templar/dataset.json ]

COPY --chown=appuser:appuser environments/templar/env.py /app/

# security_defs.py stub install (env.py imports crusades.core.security_defs)
USER root
RUN mkdir -p /app/crusades/core \
    && touch /app/crusades/__init__.py /app/crusades/core/__init__.py \
    && chown -R appuser:appuser /app/crusades
USER appuser
COPY --chown=appuser:appuser src/crusades/core/security_defs.py /app/crusades/core/security_defs.py

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV CACHED_DATASET_PATH="/home/appuser/.cache/templar/dataset.json"
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000"]
