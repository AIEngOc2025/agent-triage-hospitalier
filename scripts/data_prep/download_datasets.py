import os
from datasets import load_dataset

# Répertoire de destination
os.makedirs("data/raw", exist_ok=True)

# Liste des datasets à télécharger (identifiants Hugging Face)
# Note: Ces datasets peuvent demander une acceptation de licence sur Hugging Face.
sources = {
    "MediQA": "bigbio/mediqa",
    "MedQuAD": "lavita/medquad",
    "FrenchMedMCQA": "neuropark/french_medmcqa"
}

for name, path in sources.items():
    print(f"📥 Téléchargement de {name} depuis {path}...")
    try:
        # load_dataset télécharge le dataset
        dataset = load_dataset(path)
        # Sauvegarde en format disque Hugging Face
        dataset.save_to_disk(f"data/raw/{name}")
        print(f"✅ {name} téléchargé avec succès dans data/raw/{name}.")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de {name}: {e}")
