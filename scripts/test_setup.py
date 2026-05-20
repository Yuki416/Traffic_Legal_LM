import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="yentinglin/Llama-3-Taiwan-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

print("Environment setup complete!")
