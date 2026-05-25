FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils \
    python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip

# torch 2.12.0+cu126: latest cu126 wheel, satisfies unsloth_zoo's torch>=2.4.0,<2.13.0.
# torchvision 0.27.0 satisfies torch 2.12.0's torchvision>=0.27.0 requirement.
# Base image is cuda:12.6 (available on Docker Hub); Driver 580 on host is backward-compatible.
RUN pip install \
    torch==2.12.0 torchvision==0.27.0 \
    --index-url https://download.pytorch.org/whl/cu126

# No version constraints needed — torch 2.12.0 naturally satisfies all unsloth dependencies.
RUN pip install \
    unsloth_zoo \
    "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    datasets bitsandbytes huggingface_hub \
    wandb scipy sentencepiece protobuf

WORKDIR /workspace
