from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use the base model that matches the SFT/DPO training
base_model_id = "Qwen/Qwen2.5-0.5B"
adapter_path = "models/dpo_final_chsa"
save_path = "models/merged_dpo_final_chsa"

print("🧬 Fusion du modèle pour déploiement vLLM...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.float16, device_map="cpu"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Modèle fusionné sauvegardé dans {save_path}")
