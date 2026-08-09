import json
import os

from app.api_utils import MedicalAnonymizer


def process_nlie2_triage(input_path, output_path):
    """
    @definition : Traite le dataset de triage NLie2 (format row-based) et formate en instruction/response.
    @args/params : input_path, output_path
    @return : None
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    anonymizer = MedicalAnonymizer()
    processed_data = []

    # Mapping des zones de triage NLie2 vers nos catégories
    # 'Red' -> Maximale, 'Yellow' -> Modérée, 'Green' -> Différée, 'Black' -> Différée/Autre
    triage_map = {
        "Red": "maximale",
        "Yellow": "modérée",
        "Green": "différée",
        "Black": "différée",
    }

    for row in data.get("rows", []):
        r = row["row"]
        try:
            question = r["question"]
            zone = r["triage_zone"]

            # Mapper la zone
            urgency_category = triage_map.get(zone, "différée")

            # Formatage pour l'agent
            instruction = f"Patient avec les symptômes suivants : {question}. Quel est le niveau d'urgence ?"
            response = f"[Niveau: {urgency_category}] - Orientation : {r['action']}"

            # Anonymisation
            clean_inst = anonymizer.anonymize_text(instruction)
            clean_resp = anonymizer.anonymize_text(response)

            processed_data.append({"instruction": clean_inst, "response": clean_resp})
        except Exception:
            continue

    # Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ {len(processed_data)} exemples de NLie2 traités dans {output_path}")


if __name__ == "__main__":
    process_nlie2_triage(
        "data/raw/triage_nlie2.json", "data/processed/triage_nlie2_processed.jsonl"
    )
