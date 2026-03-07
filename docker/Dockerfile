# Templar Crusades Validator
#
# URL-Based Architecture:
# - Validator reads miner commitments from blockchain
# - Downloads miner code from committed URLs
# - Evaluates code in isolated Docker containers
# - Sets weights based on performance
#
# Security:
# - Runs as non-root user (appuser) for container security
#
# Build:
#   docker build -t templar-crusades:latest .
#
# Usage:
#   docker-compose up -d

FROM python:3.12-slim

# Install system dependencies (as root)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    docker.io \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain (needed to compile basilica-sdk)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create non-root user for security
RUN useradd -m -s /bin/bash appuser && \
    usermod -aG docker appuser

WORKDIR /app

# Install uv for faster dependency management
RUN pip install uv

# Copy dependency spec and source code (editable install needs source to exist)
COPY --chown=appuser:appuser pyproject.toml uv.lock* ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser neurons/ ./neurons/

# Install package in editable mode
RUN uv pip install --system -e .

# Copy remaining config/data files
COPY --chown=appuser:appuser hparams/ ./hparams/
COPY --chown=appuser:appuser environments/ ./environments/

# Switch to non-root user
USER appuser

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app

# Default command runs validator
CMD ["python", "-m", "neurons.validator", \
     "--wallet-name", "${WALLET_NAME}", \
     "--wallet-hotkey", "${WALLET_HOTKEY}", \
     "--netuid", "${NETUID}", \
     "--subtensor-network", "${NETWORK}", \
     "--affinetes-mode", "${AFFINETES_MODE:-docker}"]
