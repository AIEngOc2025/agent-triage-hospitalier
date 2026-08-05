import json
from pathlib import Path

# Configuration
INPUT_FILE = Path("data/processed/Mpaga_Christophe_1_Dataset_Train_DPO_052026.jsonl")
OUTPUT_FILE = Path("data/processed/Mpaga_Christophe_1_Dataset_Train_DPO_Filtered.jsonl")


def filter_dataset():
    """
    @definition : Filtre le dataset pour ne garder que les exemples de triage pur
    et les cas cliniques symptomatiques, en excluant l'encyclopédie médicale.
    @args/params : Aucune.
    @return : Aucun résultat retourné (fichier sauvegardé).
    """
    count_kept = 0

    print(f"🔄 Filtrage de {INPUT_FILE}...")

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as f_in,
        open(OUTPUT_FILE, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            data = json.loads(line)
            instr = data.get("instruction", "").lower()
            resp = data.get("response", "").lower()

            # Critères de conservation (plus larges pour le DPO)
            medical_keywords = [
                "patient",
                "medical",
                "triage",
                "symptom",
                "douleur",
                "urgent",
                "health",
                "doctor",
                "diagnos",
            ]
            instr = data.get("prompt", "").lower()

            if any(kw in instr for kw in medical_keywords):
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                count_kept += 1

    print(f"✅ Filtrage terminé. {count_kept} lignes conservées dans {OUTPUT_FILE}")


if __name__ == "__main__":
    filter_dataset()
