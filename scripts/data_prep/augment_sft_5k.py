import json
import random


def augment_data(input_path, output_path, target_count=5000):
    """
    @definition : Augmente le dataset à 5000 exemples en multipliant
    et variant les données.
    @args/params : input_path, output_path, target_count.
    @return : None.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    current_count = len(data)
    if current_count == 0:
        print("❌ Dataset vide, impossible d'augmenter.")
        return

    # Calculer le multiplicateur nécessaire
    multiplier = (target_count // current_count) + 1

    augmented_data = []

    # Stratégie : Variantes simples pour augmenter le volume sans perdre le sens
    variations = [
        "",
        " (Note : le triage est conforme aux protocoles du CHSA).",
        " (Rappel : évaluer en priorité les constantes vitales).",
        " (Action : orienter immédiatement vers la zone adéquate).",
    ]

    for _ in range(multiplier):
        for example in data:
            if len(augmented_data) >= target_count:
                break

            # Créer une variante
            new_example = example.copy()
            variation = random.choice(variations)
            new_example["response"] = example["response"] + variation
            augmented_data.append(new_example)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in augmented_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"✅ Dataset augmenté créé : {len(augmented_data)} exemples dans {output_path}"
    )


if __name__ == "__main__":
    input_file = "data/processed/train_sft.jsonl"
    output_file = "data/processed/train_sft_final_5k_triage.jsonl"
    augment_data(input_file, output_file)
