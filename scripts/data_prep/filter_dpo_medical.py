import json

# Mots-clés pour filtrer le contenu médical
MEDICAL_KEYWORDS = [
    "patient",
    "medical",
    "health",
    "doctor",
    "hospital",
    "symptom",
    "disease",
    "diagnosis",
    "treatment",
    "medicine",
    "nurse",
    "clinical",
    "surgery",
    "pain",
    "condition",
    "illness",
    "emergency",
]


def is_medical(text):
    """
    @definition : Vérifie si un texte contient des mots-clés médicaux.
    @args/params : text (str) le texte à analyser.
    @return : bool True si médical, False sinon.
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in MEDICAL_KEYWORDS)


def filter_dpo_data(input_path, output_path):
    """
    @definition : Filtre le dataset DPO pour ne garder que le contenu médical.
    @args/params : input_path (str), output_path (str).
    @return : None.
    """
    filtered_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # On vérifie si le prompt ou les réponses sont médicales
            if (
                is_medical(entry["prompt"])
                or is_medical(entry["chosen"])
                or is_medical(entry["rejected"])
            ):
                filtered_data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in filtered_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"✅ Dataset filtré créé : {len(filtered_data)} paires conservées "
        f"dans {output_path}"
    )


if __name__ == "__main__":
    input_file = "data/processed/Mpaga_Christophe_1_Dataset_DPO_Final_052026.jsonl"
    output_file = (
        "data/processed/Mpaga_Christophe_1_Dataset_DPO_Final_Medical_Only.jsonl"
    )
    filter_dpo_data(input_file, output_file)
