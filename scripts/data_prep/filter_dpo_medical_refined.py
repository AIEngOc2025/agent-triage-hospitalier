import json

# Mots-clés plus spécifiques pour le triage médical (hors <PATIENT> pour éviter les faux positifs)
MEDICAL_KEYWORDS = [
    "triage",
    "symptom",
    "emergency",
    "urgency",
    "pain",
    "medical",
    "clinical",
    "hospital",
    "diagnosis",
    "nurse",
    "doctor",
    "treatment",
    "illness",
    "condition",
]


def is_medical_refined(text):
    """
    @definition : Vérifie si un texte contient des mots-clés médicaux spécifiques.
    @args/params : text (str) le texte à analyser.
    @return : bool True si médical, False sinon.
    """
    text_lower = text.lower()
    # On vérifie les mots-clés, en ignorant le token <PATIENT>
    return any(keyword in text_lower for keyword in MEDICAL_KEYWORDS)


def filter_dpo_data_refined(input_path, output_path):
    """
    @definition : Filtre le dataset DPO avec une logique plus précise.
    @args/params : input_path (str), output_path (str).
    @return : None.
    """
    filtered_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # On vérifie prompt, chosen et rejected
            if (
                is_medical_refined(entry["prompt"])
                or is_medical_refined(entry["chosen"])
                or is_medical_refined(entry["rejected"])
            ):
                filtered_data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in filtered_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"✅ Dataset raffiné créé : {len(filtered_data)} paires conservées dans {output_path}"
    )


if __name__ == "__main__":
    # On part du fichier original non filtré pour ne pas perdre de données
    input_file = "data/processed/Mpaga_Christophe_1_Dataset_DPO_Final_052026.jsonl"
    output_file = (
        "data/processed/Mpaga_Christophe_1_Dataset_DPO_Final_Medical_Refined.jsonl"
    )
    filter_dpo_data_refined(input_file, output_file)
