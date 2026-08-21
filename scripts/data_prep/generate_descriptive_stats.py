"""Génère un rapport de statistiques détaillé pour un ou plusieurs datasets JSONL.

Ce script calcule et sauvegarde des métriques clés sur un ou plusieurs datasets, incluant :
- La distribution linguistique.
- Les métriques d'anonymisation.
- La longueur moyenne et la distribution des instructions et des réponses.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

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


def detect_lang(text: str) -> str:
    """
    @definition : Détecte la langue (fr/en) basée sur des mots outils.
    @args/params : text (str)
    @return : str ('fr' ou 'en')
    """
    fr_words = {" le ", " la ", " les ", " est ", " vous ", " dans ", " pour "}
    text_lower = " " + text.lower() + " "
    return "fr" if any(word in text_lower for word in fr_words) else "en"


def get_file_stats(file_path: Path) -> dict | None:
    """
    @definition : Calcule les stats pour un seul fichier.
    @return : dict avec les statistiques.
    """
    if not file_path.exists():
        return None

    total_rows = 0
    lang_dist = Counter()
    anon_metrics = Counter()
    instruction_lengths = []
    response_lengths = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                total_rows += 1

                instruction = data.get("instruction", "")
                response = data.get("response", "")

                # Métriques de langue et de longueur
                lang = detect_lang(instruction)
                lang_dist[lang] += 1
                instruction_lengths.append(len(instruction))
                response_lengths.append(len(response))

                # Anonymisation
                full_text = instruction + response
                for tag in ANONYMIZATION_TAGS:
                    anon_metrics[tag] += full_text.count(tag)

            except (json.JSONDecodeError, KeyError):
                continue

    if total_rows == 0:
        return None

    return {
        "dataset_name": file_path.name,
        "total_rows": total_rows,
        "language_distribution": dict(lang_dist),
        "anonymization_metrics": {k: v for k, v in anon_metrics.items() if v > 0},
        "average_lengths": {
            "instruction": round(sum(instruction_lengths) / total_rows, 2),
            "response": round(sum(response_lengths) / total_rows, 2),
        },
    }


def generate_batch_stats(directory_path: Path, output_dir: Path | None = None):
    """
    @definition : Parcourt le répertoire, génère les stats et les affiche/sauvegarde.
    """
    if not directory_path.is_dir():
        print(f"❌ Erreur : {directory_path} n'est pas un dossier.")
        return

    jsonl_files = list(directory_path.glob("*.jsonl"))
    print(f"\n📊 Analyse de {len(jsonl_files)} fichiers dans : {directory_path}\n")

    all_stats = []
    for file in sorted(jsonl_files):
        stats = get_file_stats(file)
        if stats:
            all_stats.append(stats)
            print(f"--- Statistiques pour {file.name} ---")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            print("-" * 40)

    if output_dir and all_stats:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Sauvegarde un rapport par fichier pour un suivi fin
        for report in all_stats:
            report_path = output_dir / f"{Path(report['dataset_name']).stem}_stats.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            print(f"✅ Rapport sauvegardé dans : {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stats consolidées pour datasets JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dossier contenant les fichiers .jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Dossier optionnel pour sauvegarder les rapports JSON.",
    )
    args = parser.parse_args()
    generate_batch_stats(args.input_dir, args.output_dir)
