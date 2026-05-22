import torch
from unsloth import FastLanguageModel

CASES = [
    "甲駕車行駛於市區道路，在綠燈時直行，乙騎機車闖紅燈左轉，雙方發生碰撞。甲車頭受損，乙人受傷。請問責任如何判斷？",
    "丙在高速公路行駛，因前車丁突然緊急煞車，丙來不及反應追撞丁車。請問丙、丁各應負何種責任？賠償範圍包含哪些項目？",
    "戊騎機車在夜間行駛，未開大燈，被己駕駛之汽車從後方追撞。事後發現戊有過失，應如何計算過失比例與賠償金額？",
]

print("=" * 60)
print("基線測試：Llama-3-Taiwan-8B-Instruct 原始回答品質")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="yentinglin/Llama-3-Taiwan-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

for i, case in enumerate(CASES, 1):
    print(f"\n【案例 {i}】\n{case}\n")
    messages = [{"role": "user", "content": case}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"【模型回答】\n{response}")
    print("-" * 60)

print("\n基線測試完成。")
