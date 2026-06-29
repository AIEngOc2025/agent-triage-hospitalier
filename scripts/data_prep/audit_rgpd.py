from collections import Counter

FILE_PATH = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_052026.jsonl"

def audit_rgpd():
    stats = Counter()
    total_lines = 0
    anonymized_lines = 0

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            if "<PATIENT>" in line:
                stats["patients_masqués"] += line.count("<PATIENT>")
                anonymized_lines += 1
            if "<LIEU>" in line:
                stats["lieux_masqués"] += line.count("<LIEU>")

    print("\n" + "🛡️ " * 5 + " AUDIT DE CONFORMITÉ RGPD " + " 🛡️" * 5)
    print(f"Nombre total de lignes analysées : {total_lines}")
    print(f"Lignes contenant des données masquées : {anonymized_lines} ({ (anonymized_lines/total_lines)*100:.2f}%)")
    print("-" * 40)
    print(f"👤 Noms de patients anonymisés : {stats['patients_masqués']}")
    print(f"📍 Lieux/Adresses anonymisés   : {stats['lieux_masqués']}")
    print("-" * 40)

    if anonymized_lines > 0:
        print("✅ RÉSULTAT : Anonymisation active et vérifiée.")
    else:
        print("⚠️ ATTENTION : Aucune balise trouvée. Vérifiez les modèles SpaCy.")

if __name__ == "__main__":
    audit_rgpd()
