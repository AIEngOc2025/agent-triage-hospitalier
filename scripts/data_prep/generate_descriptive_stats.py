import json
from collections import Counter
from pathlib import Path

# Configuration du chemin (Nom officiel de ton dataset)
BASE_PATH = Path("data/processed")
FILE_PATH = BASE_PATH / "train_sft_triage_only.jsonl"
REPORT_PATH = Path("reports/metrics/dataset_stats_summary.json")
ANONYMIZATION_TAGS = ["<PATIENT>", "<LIEU>", "<DATE>", "<TEL>", "<EMAIL>"]

def detect_lang(text):
    """
    @definition : Détecte la langue (fr/en) basée sur des mots outils.
    @args/params : text (str)
    @return : str ('fr' ou 'en')
    """
    fr_words = {" le ", " la ", " les ", " est ", " vous ", " dans ", " pour "}
    text_lower = " " + text.lower() + " "
    return "fr" if any(word in text_lower for word in fr_words) else "en"

def generate_stats():
    """
    @definition : Génère des statistiques descriptives sur le dataset SFT
                  (longueur, bilinguisme, anonymisation).
    @args/params : Aucun.
    @return : None.
    """
    if not FILE_PATH.exists():
        print(f"❌ Erreur : Fichier {FILE_PATH} introuvable.")
        return

    # Initialisation des compteurs
    total_rows = 0
    lang_counter = Counter()
    anonymization_stats = Counter()
    char_counts_instr = []
    char_counts_resp = []

    print(f"📊 Analyse du dataset : {FILE_PATH}...")

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                total_rows += 1

                # 1. Taille du jeu de données (Statistiques de longueur)
                instr = data.get("instruction", "")
                resp = data.get("response", "")
                char_counts_instr.append(len(instr))
                char_counts_resp.append(len(resp))

                # 2. Bilinguisme (Re-détection robuste)
                lang = detect_lang(instr)
                lang_counter[lang] += 1

                # 3. Anonymisation (Détection des tags RGPD)
                for tag in ANONYMIZATION_TAGS:
                    if tag in instr or tag in resp:
                        # On compte le nombre total d'occurrences
                        count = instr.count(tag) + resp.count(tag)
                        anonymization_stats[tag] += count

            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ Ligne {i} ignorée en raison d'une erreur: {e}")
                continue

    # Calcul des moyennes de longueur
    avg_len_instr = sum(char_counts_instr) / total_rows if total_rows > 0 else 0
    avg_len_resp = sum(char_counts_resp) / total_rows if total_rows > 0 else 0

    # Affichage du rapport (Format PDF/Jury)
    print("\n" + "═" * 60)
    print("      RAPPORT DE STATISTIQUES DESCRIPTIVES - PROJET CHSA")
    print("═" * 60)

    print("\n📈 VOLUMÉTRIE GÉNÉRALE")
    print(f"   - Nombre total de paires SFT : {total_rows:,}")
    print(f"   - Longueur moyenne Instruction: {avg_len_instr:.1f} car.")
    print(f"   - Longueur moyenne Réponse    : {avg_len_resp:.1f} car.")

    print("\n🌍 RÉPARTITION DU BILINGUISME")
    for lang, count in lang_counter.items():
        pct = (count / total_rows) * 100
        flag = "🇫🇷 FR" if lang == "fr" else "🇺🇸 EN"
        print(f"   - {flag:<10} : {count:>6} exemples ({pct:>6.2f} %)")

    print("\n🛡️  CONFORMITÉ RGPD (ANONYMISATION)")
    total_anonymized_tags = sum(anonymization_stats.values())
    print(f"   - Total d'entités masquées    : {total_anonymized_tags}")
    for tag, count in anonymization_stats.items():
        print(f"     {tag:<12} : {count:>5} remplacements")

    print("\n" + "═" * 60)

    # Sauvegarde du rapport en JSON pour l'intégrer au rapport technique
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "dataset_name": FILE_PATH.name,
        "total_rows": total_rows,
        "language_distribution": dict(lang_counter),
        "anonymization_metrics": dict(anonymization_stats),
        "average_lengths": {
            "instruction": round(avg_len_instr, 2),
            "response": round(avg_len_resp, 2),
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Rapport sauvegardé dans : {REPORT_PATH}")


if __name__ == "__main__":
    generate_stats()
