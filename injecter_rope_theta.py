import json
import os

# Chemin vers votre modèle fusionné
config_path = "models/merged_dpo_final_chsa/config.json"

if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # Ajout du paramètre manquant pour Qwen2.5 / Qwen3
    # La valeur standard pour cette architecture est 1000000.0
    if "rope_theta" not in config:
        print("🔧 Ajout du paramètre 'rope_theta' manquant...")
        config["rope_theta"] = 1000000.0

        # Par sécurité, vérifions aussi d'autres paramètres souvent requis par MLX
        if "sliding_window" not in config:
            config["sliding_window"] = None

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("✅ Fichier config.json mis à jour avec succès.")
    else:
        print(
            "ℹ️ 'rope_theta' est déjà présent. Le problème vient peut-être d'une version de mlx-lm trop récente."
        )
else:
    print(f"❌ Erreur : Fichier introuvable à l'adresse {config_path}")
