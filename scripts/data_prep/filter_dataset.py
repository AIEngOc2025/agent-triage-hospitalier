import json
from pathlib import Path

# Configuration
INPUT_FILE = Path("data/processed/train_sft_split.jsonl")
OUTPUT_FILE = Path("data/processed/train_sft_triage_only.jsonl")

def filter_dataset():
    """
    @definition : Filtre le dataset pour ne garder que les exemples de triage pur
    et les cas cliniques symptomatiques, en excluant l'encyclopédie médicale.
    @args/params : Aucune.
    @return : Aucun résultat retourné (fichier sauvegardé).
    """
    count_kept = 0
    
    print(f"🔄 Filtrage de {INPUT_FILE}...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            data = json.loads(line)
            instr = data.get("instruction", "").lower()
            resp = data.get("response", "").lower()
            
            # Critères de conservation
            is_triage_dialogue = "agis comme un assistant de triage" in instr
            is_symptom_oriented = "symptom" in instr or "symptômes" in instr or "douleur" in instr
            
            if is_triage_dialogue or is_symptom_oriented:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                count_kept += 1
                
    print(f"✅ Filtrage terminé. {count_kept} lignes conservées dans {OUTPUT_FILE}")

if __name__ == "__main__":
    filter_dataset()
