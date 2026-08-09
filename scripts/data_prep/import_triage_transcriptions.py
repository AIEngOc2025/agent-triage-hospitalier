import json
import os

from app.api_utils import MedicalAnonymizer


def process_triage_transcriptions(input_path, output_path):
    """
    @definition : Traite le dataset de triage_transcriptions (row-based) et formate en instruction/response.
    @args/params : input_path, output_path
    @return : None
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    anonymizer = MedicalAnonymizer()
    processed_data = []

    # Mapping des zones de triage vers nos catégories
    triage_map = {
        "Red": "maximale",
        "Yellow": "modérée",
        "Green": "différée",
        "Black": "différée",
    }

    for row in data.get("rows", []):
        r = row["row"]
        try:
            # Extraction
            question = r["question"]
            r["transcription"]
            zone = r["triage_zone"]

            # Mapper la zone
            urgency_category = triage_map.get(zone, "différée")

            # Formatage pour l'agent (triage + recommandation)
            instruction = f"Patient avec les symptômes suivants : {question}. Quel est le niveau d'urgence ?"
            # Utilisation du format strict demandé par le Guided Decoding
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

    print(
        f"✅ {len(processed_data)} exemples de triage_transcriptions traités dans {output_path}"
    )


if __name__ == "__main__":
    process_triage_transcriptions(
        "data/raw/triage_transcriptions.json",
        "data/processed/triage_transcriptions_processed.jsonl",
    )
