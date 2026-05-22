# Traffic Legal LM

**中文** | [English](#english)

## 專案簡介

以 [Unsloth](https://github.com/unslothai/unsloth) 框架對 [Llama-3-Taiwan-8B-Instruct](https://huggingface.co/yentinglin/Llama-3-Taiwan-8B-Instruct) 進行 LoRA 微調，任務為台灣車禍交通事故的法律文件分析。

## 功能目標

- 給定車禍案例描述，自動分析責任歸屬
- 引用相關法條（道路交通管理處罰條例、民法等）
- 估算賠償範圍與項目

## 環境需求

- NVIDIA GPU（建議 VRAM ≥ 16GB）
- Docker + NVIDIA Container Toolkit
- CUDA 12.1+

## 快速開始

**1. Build 環境**
```bash
docker build -t legal-lora:latest .
```

**2. 驗證環境**
```bash
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  legal-lora:latest python scripts/test_setup.py
```

**3. 執行測試**
```bash
# GPU 記憶體用量
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  legal-lora:latest python scripts/test_gpu_memory.py

# LoRA 訓練流程（假資料）
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  legal-lora:latest python scripts/test_lora_dummy.py

# 基線測試（Base model 回答品質）
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  legal-lora:latest python scripts/test_baseline.py
```

## 目錄結構

```
Traffic_Legal_LM/
├── data/
│   ├── raw/          # 原始資料（不納入版本控制）
│   ├── processed/    # 處理後資料
│   └── synthetic/    # 合成資料
├── scripts/          # 訓練、測試腳本
├── models/           # 模型 checkpoint（不納入版本控制）
├── eval/             # 評估結果
├── demo/             # 展示用程式
└── Dockerfile
```

## 技術棧

| 項目 | 版本 |
|------|------|
| Base Model | Llama-3-Taiwan-8B-Instruct |
| Fine-tuning | Unsloth + LoRA (r=16) |
| Quantization | 4-bit (bitsandbytes) |
| Training | TRL SFTTrainer |
| GPU | NVIDIA RTX 4090 x2 |

---

<a name="english"></a>

# Traffic Legal LM

**[中文](#專案簡介)** | English

## Overview

Fine-tuning [Llama-3-Taiwan-8B-Instruct](https://huggingface.co/yentinglin/Llama-3-Taiwan-8B-Instruct) with LoRA using the [Unsloth](https://github.com/unslothai/unsloth) framework for Taiwan traffic accident legal document analysis.

## Goals

- Analyze liability given a traffic accident description
- Reference relevant laws (Road Traffic Management and Penalty Act, Civil Code, etc.)
- Estimate compensation items and amounts

## Requirements

- NVIDIA GPU (VRAM ≥ 16GB recommended)
- Docker + NVIDIA Container Toolkit
- CUDA 12.1+

## Quick Start

**1. Build the environment**
```bash
docker build -t legal-lora:latest .
```

**2. Verify the environment**
```bash
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  legal-lora:latest python scripts/test_setup.py
```

**3. Run tests**
```bash
# GPU memory profiling
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  legal-lora:latest python scripts/test_gpu_memory.py

# LoRA training smoke test (dummy data)
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  legal-lora:latest python scripts/test_lora_dummy.py

# Baseline evaluation (base model quality)
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  legal-lora:latest python scripts/test_baseline.py
```

## Project Structure

```
Traffic_Legal_LM/
├── data/
│   ├── raw/          # Raw data (excluded from version control)
│   ├── processed/    # Processed data
│   └── synthetic/    # Synthetic data
├── scripts/          # Training and testing scripts
├── models/           # Model checkpoints (excluded from version control)
├── eval/             # Evaluation results
├── demo/             # Demo application
└── Dockerfile
```

## Tech Stack

| Component | Detail |
|-----------|--------|
| Base Model | Llama-3-Taiwan-8B-Instruct |
| Fine-tuning | Unsloth + LoRA (r=16) |
| Quantization | 4-bit (bitsandbytes) |
| Training | TRL SFTTrainer |
| GPU | NVIDIA RTX 4090 x2 |
