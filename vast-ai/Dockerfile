FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install numpy && \
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace

