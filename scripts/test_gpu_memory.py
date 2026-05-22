import torch
from unsloth import FastLanguageModel

def report_vram():
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        used  = torch.cuda.memory_allocated(i) / 1024**3
        free  = total - used
        print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): "
              f"used={used:.2f}GB / total={total:.2f}GB / free={free:.2f}GB")

print("=" * 60)
print("GPU 記憶體用量測試")
print("=" * 60)

print("\n[載入前]")
report_vram()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="yentinglin/Llama-3-Taiwan-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

print("\n[4-bit 量化模型載入後]")
report_vram()

# Apply LoRA to measure overhead
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print("\n[套用 LoRA adapter 後]")
report_vram()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nLoRA 可訓練參數: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print("\nGPU 記憶體測試完成。")
