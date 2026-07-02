import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-1.5B"
ADAPTERS = "models/sft_final_chsa"

# 1. On charge sur le CPU pour être certain de la stabilité
device = "cpu" 
print(f"🔄 Test diagnostic sur {device.upper()}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float32, 
    device_map={"": device}
)
model = PeftModel.from_pretrained(base_model, ADAPTERS)

# 2. On recrée le format EXACT de l'entraînement
messages = [
    {"role": "system", "content": "Tu es l'infirmier d'accueil du CHSA."},
    {"role": "user", "content": "Bonjour, j'ai une douleur à la poitrine."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)

# 3. Génération TRÈS stricte
print("⚡ Génération...")
outputs = model.generate(
    **inputs, 
    max_new_tokens=50, 
    do_sample=False, 
    repetition_penalty=1.5  # On force le modèle à NE PAS répéter le même caractère
)

print("\nAGENT IA :")
print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1])
