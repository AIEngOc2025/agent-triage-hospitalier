import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DPO_ADAPTERS = "models/dpo_final_chsa" # Chemin vers tes poids DPO

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                dtype=torch.float32,
                                                device_map={"": "mps"}
                                                    )

model = PeftModel.from_pretrained(base_model, DPO_ADAPTERS)

query = "J'ai une douleur vive au bras gauche et je transpire."
prompt = f"<|im_start|>system\nTu es l'infirmier du CHSA.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to("mps")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, repetition_penalty=1.2)

print("\n🚀 RÉPONSE FINALE DE L'AGENT (POST-DPO) :")
print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip())
