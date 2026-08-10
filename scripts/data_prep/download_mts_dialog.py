import os

from datasets import load_dataset

# Répertoire de destination
os.makedirs("data/raw", exist_ok=True)
url = "https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note"
print("📥 Téléchargement de MTS-Dialog depuis Hugging Face...")
try:
    # Le dataset d'origine de MTS-Dialog est souvent sous "Lekunb/mts-dialog"
    # ou disponible via d'autres repos
    dataset = load_dataset(url)
    dataset.save_to_disk("data/raw/mts-dialog")
    print("✅ MTS-Dialog téléchargé avec succès dans data/raw/mts-dialog.")
except Exception as e:
    print(f"❌ Erreur directe, tentative avec un nom alternatif... {e}")
    try:
        # Autre tentative de nom si le premier échoue
        dataset = load_dataset("lavita/mts-dialog")
        dataset.save_to_disk("data/raw/mts-dialog")
        print("✅ MTS-Dialog téléchargé avec succès (via lavita/mts-dialog).")
    except Exception as e2:
        print(f"❌ Échec de la deuxième tentative : {e2}")
