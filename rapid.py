import json
import os
import shutil

# --- CONFIGURATION ---
model_path = os.path.abspath("models/merged_final_dpo_chsa")
config_file = os.path.join(model_path, "tokenizer_config.json")

print(f"🔧 Analyse du dossier : {model_path}")

if os.path.exists(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 1. Force la classe Fast (Rust)
    data["tokenizer_class"] = "Qwen2TokenizerFast"
    
    # 2. Ajoute les attributs manquants que vLLM cherche
    data["added_tokens_decoder"] = data.get("added_tokens_decoder", {})
    data["clean_up_tokenization_spaces"] = False
    
    # 3. Sauvegarde le fichier corrigé
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("✅ tokenizer_config.json mis à jour.")

    # 4. Vérification du fichier binaire tokenizer.json
    # Si ce fichier manque, vLLM ne pourra jamais passer en mode "Fast"
    if not os.path.exists(os.path.join(model_path, "tokenizer.json")):
        print("⚠️ Fichier tokenizer.json manquant ! Tentative de récupération...")
        # On essaie de le copier depuis le cache si possible, ou on conseille de le télécharger
        print("💡 CONSEIL : Copiez le fichier 'tokenizer.json' du modèle Qwen2.5-0.5B original dans ce dossier.")
else:
    print("❌ Erreur : Fichier tokenizer_config.json introuvable.")