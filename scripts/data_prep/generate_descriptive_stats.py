import json
import os
from collections import Counter

# Configuration du chemin (Nom officiel de ton dataset)
FILE_PATH = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_052026.jsonl"
REPORT_PATH = "reports/metrics/dataset_stats_summary.json"

def generate_stats():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Erreur : Fichier {FILE_PATH} introuvable.")
        return

    # Initialisation des compteurs
    total_rows = 0
    lang_counter = Counter()
    anonymization_stats = Counter()
    char_counts_instr = []
    char_counts_resp = []

    print(f"📊 Analyse du dataset : {FILE_PATH}...")

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                total_rows += 1

                # 1. Taille du jeu de données (Statistiques de longueur)
                instr = data.get('instruction', '')
                resp = data.get('response', '')
                char_counts_instr.append(len(instr))
                char_counts_resp.append(len(resp))

                # 2. Bilinguisme (Via les métadonnées créées précédemment)
                lang = data.get('clinical_metadata', {}).get('language', 'inconnu')
                lang_counter[lang] += 1

                # 3. Anonymisation (Détection des tags RGPD)
                for tag in ["<PATIENT>", "<LIEU>", "<DATE>", "<TEL>", "<EMAIL>"]:
                    if tag in instr or tag in resp:
                        # On compte le nombre total d'occurrences
                        count = instr.count(tag) + resp.count(tag)
                        anonymization_stats[tag] += count

            except Exception:
                continue

    # Calcul des moyennes de longueur
    avg_len_instr = sum(char_counts_instr) / total_rows if total_rows > 0 else 0
    avg_len_resp = sum(char_counts_resp) / total_rows if total_rows > 0 else 0

    # Affichage du rapport (Format PDF/Jury)
    print("\n" + "═"*60)
    print("      RAPPORT DE STATISTIQUES DESCRIPTIVES - PROJET CHSA")
    print("═"*60)

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

    print("\n" + "═"*60)

    # Sauvegarde du rapport en JSON pour l'intégrer au rapport technique
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report_data = {
        "dataset_name": os.path.basename(FILE_PATH),
        "total_rows": total_rows,
        "language_distribution": dict(lang_counter),
        "anonymization_metrics": dict(anonymization_stats),
        "average_lengths": {
            "instruction": round(avg_len_instr, 2),
            "response": round(avg_len_resp, 2)
        }
    }

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Rapport sauvegardé dans : {REPORT_PATH}")

if __name__ == "__main__":
    generate_stats()
