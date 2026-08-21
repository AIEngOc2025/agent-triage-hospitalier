import json
import random


def augment_to_target(input_path, output_path, target_count=2000):
    """
    @definition : Augmente un dataset à un nombre cible d'exemples.
    @args/params : input_path, output_path, target_count
    @return : None
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if not data:
        print(f"❌ {input_path} est vide.")
        return

    augmented_data = []
    variations = [
        "",
        " (Protocole CHSA : maintenir le patient sous surveillance).",
        " (Recommandation : évaluer les constantes vitales toutes les 15 min).",
        " (Note : assurer la traçabilité de cette intervention).",
    ]

    # Remplissage par répétition et variation
    while len(augmented_data) < target_count:
        for example in data:
            if len(augmented_data) >= target_count:
                break

            new_example = example.copy()
            variation = random.choice(variations)
            # Ajouter la variation uniquement si elle n'est pas déjà présente
            if variation and variation not in new_example["response"]:
                new_example["response"] = example["response"] + variation
            augmented_data.append(new_example)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in augmented_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ {len(augmented_data)} exemples créés dans {output_path}")


def merge_datasets(input_paths, output_path, total_target=5000):
    """
    @definition : Fusionne plusieurs datasets et limite à total_target.
    @args/params : input_paths, output_path, total_target
    @return : None
    """
    final_data = []
    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                final_data.append(json.loads(line))

    # Mélanger
    random.shuffle(final_data)

    # Limiter
    final_data = final_data[:total_target]

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"🚀 Dataset final fusionné : {len(final_data)} exemples dans {output_path}")


if __name__ == "__main__":
    sources = [
        (
            "data/processed/new_triage_data.jsonl",
            "data/processed/temp_triage_500_2k.jsonl",
        ),
    ]

    for src, dst in sources:
        augment_to_target(src, dst, 2000)

    merge_datasets(
        [s[1] for s in sources], "data/processed/train_sft_final_5k_v2.jsonl", 5000
    )
