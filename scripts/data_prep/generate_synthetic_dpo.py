import json
import random

# Cette fonction simule une génération synthétique. 
# En production, vous feriez un appel à une API (OpenAI, Anthropic, ou un modèle local plus gros).
def enrich_example(example):
    """
    @definition : Crée une variante plus détaillée d'un exemple DPO.
    @args/params : example (dict) contenant prompt, chosen, rejected.
    @return : dict exemple enrichi.
    """
    # Ici, nous créons une variante synthétique simple pour démonstration.
    # Pour un vrai enrichissement, il faudrait passer par un LLM.
    enriched = example.copy()
    enriched["chosen"] = f"{example['chosen']} (Note clinique détaillée : le triage a été effectué selon les protocoles du CHSA, en tenant compte des antécédents du <PATIENT>)."
    enriched["rejected"] = f"{example['rejected']} (Réponse trop brève, ne respectant pas les protocoles de traçabilité.)"
    return enriched

def generate_synthetic_dataset(input_path, output_path, multiplier=3):
    """
    @definition : Génère un dataset enrichi à partir des exemples existants.
    @args/params : input_path, output_path, multiplier (combien de variantes par exemple).
    @return : None.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    synthetic_data = []
    for example in data:
        synthetic_data.append(example) # Garder l'original
        for _ in range(multiplier):
            synthetic_data.append(enrich_example(example))
            
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in synthetic_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"✅ Dataset synthétique créé : {len(synthetic_data)} paires dans {output_path}")

if __name__ == "__main__":
    input_file = "data/processed/Mpaga_Christophe_1_Dataset_DPO_Final_Medical_Refined.jsonl"
    output_file = "data/processed/Mpaga_Christophe_1_Dataset_DPO_Final_Medical_Enriched.jsonl"
    generate_synthetic_dataset(input_file, output_file, multiplier=5)
