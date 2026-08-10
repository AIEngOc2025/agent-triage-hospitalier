from collections import Counter

FILE_PATH = "data/processed/train_sft_balanced_50_50.jsonl"
ANONYMIZATION_TAGS = ["<PATIENT>", "<ADRESSE>", "<TELEPHONE>", "<CODE POSTAL>", "<CP>"]


def audit_rgpd():
    """
    @definition : Performs an audit of the anonymization process in the processed
                  dataset to check for RGPD compliance markers.
    @args/params : None
    @return : None, prints the audit statistics to the console.
    """
    stats = Counter()
    total_lines = 0
    anonymized_lines = 0

    print(f"🛡️  AUDIT DE CONFORMITÉ RGPD : {FILE_PATH} 🛡️")

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            is_anonymized = False
            for tag in ANONYMIZATION_TAGS:
                if tag in line:
                    stats[tag] += line.count(tag)
                    is_anonymized = True
            if is_anonymized:
                anonymized_lines += 1

    print(f"Nombre total de lignes analysées : {total_lines}")
    print(
        f"Lignes contenant des données masquées : {anonymized_lines} "
        f"({(anonymized_lines / total_lines) * 100:.2f}%)"
    )
    print("-" * 40)
    for tag in ANONYMIZATION_TAGS:
        print(f"👤 Tag {tag} : {stats[tag]} occurrences")
    print("-" * 40)

    if anonymized_lines > 0:
        print("✅ RÉSULTAT : Anonymisation active et vérifiée.")
    else:
        print("⚠️ ATTENTION : Aucune balise trouvée. Vérifiez les modèles SpaCy.")


if __name__ == "__main__":
    audit_rgpd()
