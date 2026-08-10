from datasets import Features, Value, load_dataset


def convert_to_hf_format(input_path: str, output_path: str):
    """
    @definition : Convertit un fichier .jsonl au format Hugging Face
    Dataset (Arrow) avec un schéma de features strict.
    @args/params :
        - input_path (str): Chemin du fichier .jsonl d'entrée.
        - output_path (str): Chemin du répertoire de sauvegarde au format Arrow.
    @return : None
    """
    # Définition du schéma strict pour valider les données
    sft_features = Features(
        {"instruction": Value("string"), "response": Value("string")}
    )

    print(f"🔄 Chargement du dataset depuis {input_path}...")

    # Chargement avec validation des features
    ds = load_dataset(
        "json", data_files=input_path, features=sft_features, split="train"
    )

    # Sauvegarde au format Arrow
    ds.save_to_disk(output_path)

    print(f"✅ Dataset converti et sauvegardé dans : {output_path}")
    print(f"   Nombre d'exemples : {len(ds)}")


if __name__ == "__main__":
    INPUT_FILE = "data/processed/train_sft_final_5k_triage.jsonl"
    OUTPUT_DIR = "data/processed/hf_dataset_triage"

    convert_to_hf_format(INPUT_FILE, OUTPUT_DIR)
