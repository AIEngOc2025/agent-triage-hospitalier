import argparse
import json
from collections import Counter
from pathlib import Path

# Configuration par défaut
ANONYMIZATION_TAGS = [
    "<PATIENT>",
    "<LIEU>",
    "<DATE>",
    "<TEL>",
    "<EMAIL>",
    "<ADRESSE>",
    "<TELEPHONE>",
    "<CODE POSTAL>",
    "<CP>",
]


def detect_lang(text):
    """
    @definition : Détecte la langue (fr/en) basée sur des mots outils.
    @args/params : text (str)
    @return : str ('fr' ou 'en')
    """
    fr_words = {" le ", " la ", " les ", " est ", " vous ", " dans ", " pour "}
    text_lower = " " + text.lower() + " "
    return "fr" if any(word in text_lower for word in fr_words) else "en"


def get_file_stats(file_path):
    """
    @definition : Calcule les stats pour un seul fichier.
    @return : dict avec les statistiques.
    """
    stats = {
        "rows": 0,
        "lang": Counter(),
        "tags": 0,
    }

    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                stats["rows"] += 1
                instr = data.get("instruction", "")

                # Bilinguisme
                stats["lang"][detect_lang(instr)] += 1

                # Anonymisation
                for tag in ANONYMIZATION_TAGS:
                    if tag in instr or tag in data.get("response", ""):
                        stats["tags"] += instr.count(tag) + data.get(
                            "response", ""
                        ).count(tag)
            except (json.JSONDecodeError, KeyError):
                continue
    return stats


def generate_batch_stats(directory_path):
    """
    @definition : Parcourt le répertoire et affiche un tableau consolidé.
    """
    path = Path(directory_path)
    if not path.is_dir():
        print(f"❌ Erreur : {directory_path} n'est pas un dossier.")
        return

    jsonl_files = list(path.glob("*.jsonl"))
    print(f"\n📊 Analyse de {len(jsonl_files)} fichiers dans : {directory_path}\n")

    # En-tête du tableau
    header = f"{'Dataset':<40} | {'Paires':>8} | {'FR %':>6} | {'EN %':>6} | {'Tags'}"
    print(header)
    print("-" * len(header))

    for file in sorted(jsonl_files):
        stats = get_file_stats(file)
        if not stats or stats["rows"] == 0:
            continue

        fr_pct = (stats["lang"]["fr"] / stats["rows"]) * 100
        en_pct = (stats["lang"]["en"] / stats["rows"]) * 100

        print(
            f"{file.name:<40} | {stats['rows']:>8} | {fr_pct:>5.1f}% | {en_pct:>5.1f}% | {stats['tags']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stats consolidées pour datasets JSONL."
    )
    parser.add_argument(
        "dir_path", type=Path, help="Dossier contenant les fichiers .jsonl"
    )
    args = parser.parse_args()
    generate_batch_stats(args.dir_path)
