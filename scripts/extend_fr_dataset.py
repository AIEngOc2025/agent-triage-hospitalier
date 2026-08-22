"""Étend le dataset avec des cas français synthétiques.

Les sources NLIE2/new_triage dominante EN produisent un dataset
quasi-exclusivement anglais. Ce script ajoute des cas FR réalistes
pour atteindre 40% de français dans le dataset d'entraînement.

Usage :
    python scripts/extend_fr_dataset.py \\
        --input data/processed/augmented_labeled.jsonl \\
        --output data/processed/augmented_labeled.jsonl \\
        --target-fr 400
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Cas français réalistes par niveau (synthétiques, équilibrés)
FR_EXAMPLES = {
    "maximale": [
        "Douleur thoracique intense irradiant dans le bras gauche, sueurs froides",
        "Détresse respiratoire aiguë, le patient ne peut plus parler",
        "Perte de connaissance brutale, patient inconscient",
        "Hémorragie massive, plaie qui ne s'arrête pas de saigner",
        "Difficulté à parler avec paralysie faciale unilatérale, suspicion d'AVC",
        "Réaction allergique sévère avec gonflement de la gorge",
        "Convulsions en cours, patient de 3 ans en état de mal épileptique",
        "Douleur thoracique avec essoufflement et tachycardie",
        "Patient polytraumatisé, chute de 4 mètres, inconscient",
        "Brûlure étendue du second degré sur 30% du corps",
        "Tachycardie à 180 battements par minute, patient agité",
        "Saignement abondant après morsure de chien au visage",
        "Céphalée brutale en coup de tonnerre, le pire mal de tête de ma vie",
        "Crise d'asthme sévère, le patient n'arrive plus à expirer",
        "Tentative de suicide par ingestion de médicaments, patient confus",
        "Douleur abdominale aiguë avec lipothymie chez une femme enceinte",
        "Arrêt cardio-respiratoire, réanimation en cours",
        "Corps étranger dans les voies aériennes, patient cyanosé",
        "Hémorragie digestive, melena et chute de tension",
        "Accident vasculaire cérébral confirmé, hémiplégie droite",
        "Embolie pulmonaire suspectée, douleur thoracique et désaturation",
        "Rupture d'anévrisme aortique, douleur abdominale intense et collapsus",
        "Pneumothorax compressif, murmure vésiculaire aboli",
        "État de choc septique, fièvre à 41 et marbrures",
        "Coma hypoglycémique, glycémie capillaire à 0.3 g/L",
        "Fracture ouverte du fémur avec saignement artériel",
        "Traumatisme crânien grave avec otorragie",
        "Tétanos, contracture généralisée et trismus",
        "Méningite à méningocoque, purpura fulminans",
        "Crise hypertensive à 220/120, signes neurologiques",
    ],
    "modérée": [
        "Fièvre à 39 depuis 48 heures avec toux productive",
        "Douleur abdominale persistante, pas de fièvre, pas de diarrhée",
        "Traumatisme du poignet après chute, déformation visible",
        "Otalgie sévère avec écoulement purulent depuis 24h",
        "Brûlure du second degré superficiel à la main droite",
        "Vomissements répétés depuis hier soir, intolérance alimentaire",
        "Crise d'asthme modérée, réponse partielle au bronchodilatateur",
        "Lombosciatique aiguë après effort, irradiation dans la jambe",
        "Plaie profonde au genou nécessitant une suture",
        "Urticaire généralisée avec prurit, sans gêne respiratoire",
        "Fièvre à 38.5 chez un enfant de 4 ans, conscient et hydraté",
        "Douleur thoracique atypique, reproductible à la palpation",
        "Vertiges depuis ce matin, instabilité à la marche",
        "Diarrhée aiguë depuis 3 jours, crampes abdominales",
        "Migraine avec aura visuelle, photophobie",
        "Crise d'angoisse avec sensation de mort imminente",
        "Tendinite du coude, impotence fonctionnelle",
        "Cystite aiguë chez une femme jeune, dysurie et pollakiurie",
        "Douleur testiculaire unilatérale brutale, sans fièvre",
        "Douleur thoracique post-traumatique, côte fêlée suspectée",
        "Pneumopathie communautaire, toux et expectorations",
        "Colique néphrétique, douleur lombaire unilatérale",
        "Sinusite maxillaire aiguë, céphalées et écoulement nasal",
        "Gastro-entérite aiguë, nausées et vomissements",
        "Réaction allergique cutanée au médicament, sans atteinte respiratoire",
        "Douleur dentaire aiguë avec abcès gingival",
        "Entorse de cheville modérée, œdème et ecchymose",
        "Infection urinaire basse chez un homme, brûlures mictionnelles",
        "Pharyngite érythémateuse, odynophagie et fièvre modérée",
        "Conjonctivite aiguë, œil rouge avec sécrétions",
    ],
    "différée": [
        "Toux sèche depuis 5 jours, pas de fièvre, état général conservé",
        "Rhinite saisonnière, éternuements et écoulement nasal clair",
        "Douleur lombaire chronique, demande de renouvellement d'arrêt",
        "Petite coupure au doigt, cicatrisation en cours",
        "Douleur articulaire modérée du genou depuis une semaine",
        "Renouvellement de traitement antihypertenseur",
        "Constipation chronique, pas d'amélioration avec les mesures hygiéno-diététiques",
        "Insomnie depuis un mois, demande de prise en charge",
        "Eczéma atopique en poussée légère, traitement habituel",
        "Fatigue persistante depuis 3 semaines, bilan à programmer",
        "Acné du visage en aggravation, demande de traitement",
        "Maux de tête ponctuels en fin de journée, soulagés par paracétamol",
        "Brûlures d'estomac après les repas, répondant aux IPP",
        "Douleur chronique de l'épaule, demande de séances de kiné",
        "Allergie au pollen habituelle, traitement antihistaminique",
        "Verrue plantaire à enlever, consultation programmée",
        "Suivi de diabète de type 2, bilan sanguin trimestriel",
        "Renouvellement de lunettes, vue floue de loin",
        "Bilan lipidique de contrôle, pas de symptôme",
        "Éruption cutanée au bras, pas de prurit, pas de fièvre",
        "Migraine occasionnelle, 2 épisodes par mois",
        "Douleur chronique du dos, kiné habituelle",
        "Eczéma des mains en hiver, crème hydratante",
        "Rhume banal, écoulement nasal et éternuements",
        "Demande de certificat médical d'aptitude sportive",
        "Vaccination à mettre à jour, pas de symptôme",
        "Aérophagie et ballonnements, améliorer le transit",
        "Aphte buccal isolé, evolution depuis 4 jours",
        "Syndrome prémenstruel sévère, demande de traitement",
        "Suivi de l'hypothyroïdie, bilan TSH",
    ],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/augmented_labeled.jsonl")
    p.add_argument("--output", default="data/processed/augmented_labeled.jsonl")
    p.add_argument(
        "--target-fr",
        type=int,
        default=400,
        help="Nombre cible de cas français dans le dataset final",
    )
    p.add_argument(
        "--variants", type=int, default=3, help="Nombre de variantes par cas de base"
    )
    args = p.parse_args()

    records = [json.loads(line) for line in open(args.input) if line.strip()]
    print(f"Input : {len(records)} cas")

    # Compte FR actuels
    def is_fr(text):
        return any(c in text for c in "éèêàùçôî") or any(
            w in text.lower()
            for w in [
                "le patient",
                "douleur",
                "fièvre",
                "présente",
                "vomissements",
                "céphalées",
                "patient avec",
                "depuis",
                "depuis",
                "depuis",
            ]
        )

    n_fr = sum(1 for r in records if is_fr(r["text"]))
    print(f"FR existants : {n_fr}")

    # Génère et ajoute des variantes FR
    rng = random.Random(42)
    records_set = {(r["text"], r["niveau"]) for r in records}
    added = 0
    new_records = []

    for niveau, exemples in FR_EXAMPLES.items():
        base = list(exemples)
        rng.shuffle(base)
        for ex in base:
            # Ajoute l'original
            key = (ex, niveau)
            if key not in records_set:
                new_records.append({"text": ex, "niveau": niveau})
                records_set.add(key)
                added += 1
            # Variantes : légères reformulations
            for v in range(args.variants):
                # Reformulation simple : variations de structure
                variants = [
                    ex,
                    ex.replace(",", " et"),
                    "Le patient : " + ex.lower(),
                    "Ce jour : " + ex,
                    "Plaintes : " + ex.lower(),
                    ex + " Pas d'amélioration.",
                    "Depuis hier, " + ex.lower(),
                    "Patient rapporte : " + ex.lower(),
                ]
                v_text = rng.choice(variants)
                if v_text != ex:
                    v_key = (v_text, niveau)
                    if v_key not in records_set:
                        new_records.append({"text": v_text, "niveau": niveau})
                        records_set.add(v_key)
                        added += 1
            if added >= args.target_fr:
                break

    # Combine et shuffle
    final = records + new_records
    rng.shuffle(final)

    # Vérif ratio
    n_fr_new = sum(1 for r in final if is_fr(r["text"]))
    print(f"Ajoutés : {added} cas FR")
    print(f"Total : {len(final)} cas")
    print(f"FR : {n_fr_new} ({100 * n_fr_new / len(final):.1f}%)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
