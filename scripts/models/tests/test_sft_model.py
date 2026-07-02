import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURATION ---
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
SFT_ADAPTERS = "models/sft"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def generate_response(query):
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # 2. Modèle de base
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        device_map={"": DEVICE},
        trust_remote_code=True
    )
    
    # 3. Adaptateurs SFT
    model = PeftModel.from_pretrained(model, SFT_ADAPTERS)
    model.eval()

    # 4. FORMAT CHATML (Celui utilisé dans ton build_dataset.py)
    # C'est ici que la magie opère : on recrée la structure apprise
    messages = [
        {"role": "system", "content": "Tu es l'infirmier d'accueil bienveillant du CHSA."},
        {"role": "user", "content": query}
    ]
    
    # On utilise la fonction de Qwen pour formater le texte proprement
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    print("⚡ Génération de la réponse médicale...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            do_sample=False, # Stable sur Mac
            repetition_penalty=1.2, # Empêche les !!!!!!!!!!!
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # On nettoie pour n'afficher que la réponse de l'infirmier
    if "assistant" in full_text:
        return full_text.split("assistant")[-1].strip()
    return full_text

if __name__ == "__main__":
    test_fr = "Bonjour, j'ai une douleur très vive dans la poitrine et mon bras gauche est engourdi. Que faut-il faire ?"
    
    print("\n" + "🩺 TEST TRIAGE CHSA " + "="*30)
    print(f"PATIENT : {test_fr}")
    response = generate_response(test_fr)
    print(f"AGENT IA : {response}")
    print("="*50 + "\n")