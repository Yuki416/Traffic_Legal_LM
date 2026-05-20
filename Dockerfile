FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch for CUDA 12.1
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install \
    transformers==4.44.2 \
    datasets==2.21.0 \
    accelerate==0.33.0 \
    peft==0.12.0 \
    trl==0.10.1 \
    bitsandbytes==0.43.3 \
    huggingface_hub==0.24.6 \
    wandb==0.17.8 \
    scipy \
    sentencepiece \
    protobuf

# Install Unsloth for CUDA 12.1 with PyTorch 2.3
RUN pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

WORKDIR /workspace
