"""Génère un dataset de triage synthétique pour démonstration.

Le LLM en prod produit trop rarement la balise `[URGENCE ...]` (le
DPO a dérivé vers un style conversationnel). Pour permettre de tester
le pipeline NLP sans dépendre du LLM, ce script produit des exemples
synthétiques et réalistes, étalonnés sur la grammaire du dataset
source.

Format identique à `label_triage_data.py` :
    {"text": "...", "niveau": "...", "raison": "...", "llm_confidence": 0.85}

Usage :
    python scripts/generate_demo_dataset.py --output data/processed/labeled_triage.jsonl --size 300
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --- Verbatim : phrases patient réalistes, par niveau ---

MAXIMALE_PHRASES = [
    "Je n'arrive plus à respirer, j'ai la poitrine qui serre",
    "Douleur thoracique intense qui irradie dans le bras gauche",
    "J'ai perdu connaissance il y a 10 minutes après un malaise",
    "Mon père ne parle plus, il a la bouche de travers, je crois qu'il fait un AVC",
    "Hémorragie abondante, ma blessure au bras ne s'arrête pas de saigner",
    "Je suis couvert de boutons et ma gorge se ferme, je ne peux plus avaler",
    "Mon fils de 3 ans a une fièvre à 40 et des convulsions",
    "J'ai fait une chute de 3 mètres, je ne sens plus mes jambes",
    "Crise d'asthme sévère, mon inhalateur ne fait plus effet",
    "Douleur pelvienne aiguë avec perte de connaissance, je suis une femme enceinte",
    "Brûlure étendue au deuxième degré sur tout le bras",
    "Je ne vois plus d'un œil depuis ce matin, c'est apparu brutalement",
    "Maux de tête brutaux et intenses, le pire de ma vie",
    "Mon ami a avalé de la javel, il est conscient mais vomit",
    "Tentative de suicide par médication, il est présent mais confuse",
    "Douleur thoracique avec sueurs froides et essoufflement",
    "Tachycardie à 180, je suis essoufflé au repos",
    "Saignement de nez abondant qui ne s'arrête pas depuis 30 minutes",
    "Mon bébé de 8 mois a une diarrhée sanglante et refuse de boire",
    "J'ai été mordu par un chien au visage, plaie ouverte",
]

MODEREE_PHRASES = [
    "Fièvre à 38.5 depuis 2 jours, je tousse beaucoup",
    "Douleur abdominale persistante depuis hier soir, pas de fièvre",
    "Je suis tombé et j'ai mal au poignet, je n'arrive pas à le bouger",
    "Maux de gorge intenses depuis 3 jours avec difficulté à avaler",
    "Otalgie sévère à droite depuis hier, écoulement jaunâtre",
    "Brûlure du second degré superficiel à la main, surface de la paume",
    "Vomissements répétés depuis 24h, je n'arrive plus à m'hydrater",
    "Crise d'asthme légère, j'ai utilisé mon inhalateur mais ça ne passe pas",
    "Douleur dorsale aiguë après un effort, irradiation dans la jambe",
    "Plaie ouverte au genou qui ne s'arrête pas de saigner depuis 20 minutes",
    "Urticaire généralisée avec démangeaisons, pas de gêne respiratoire",
    "Fièvre à 39 chez mon enfant de 5 ans, il est fatigué mais conscient",
    "Douleur thoracique légère à l'inspiration profonde depuis 2 jours",
    "Vertiges depuis ce matin, je tiens debout mais c'est instable",
    "Diarrhée aiguë depuis 3 jours avec crampes abdominales",
    "Migraine ophtalmique avec aura visuelle, je vois scintiller",
    "Crise d'angoisse importante avec sensation de mort imminente",
    "Tendinite du coude invalidante, je ne peux plus porter d'objet",
    "Cystite aiguë avec brûlures mictionnelles et fièvre à 38",
    "Douleur testiculaire brutale, unilatérale, sans fièvre",
]

DIFFEREE_PHRASES = [
    "Petite toux sèche depuis 3 jours, pas de fièvre",
    "Rhinite saisonnière, éternuements et écoulement nasal clair",
    "Mal de dos chronique, je voudrais un renouvellement d'arrêt",
    "Petite coupure au doigt, ça ne saigne plus mais je voudrais vérifier",
    "Douleur articulaire modérée au genou depuis 1 semaine",
    "Consultation pour renouvellement de traitement antihypertenseur",
    "Constipation chronique qui ne s'améliore pas",
    "Insomnie depuis 1 mois, je voudrais un somnifère",
    "Eczéma atopique en poussée légère, je connais mes traitements",
    "Fatigue persistante depuis 3 semaines, pas d'autres symptômes",
    "Acné du visage en aggravation, j'aimerais un traitement",
    "Maux de tête ponctuels en fin de journée, soulagés par paracétamol",
    "Brûlures d'estomac après les repas, allégées par les pansements",
    "Douleur chronique de l'épaule, je voudrais des séances de kiné",
    "Allergie au pollen habituelle, je connais mes antihistaminiques",
    "Verrue plantaire douloureuse à la marche, je consulte pour ablation",
    "Suivi de mon diabète, demande de bilan sanguin",
    "Renouvellement de lunettes, vue floue de loin",
    "Bilan sanguin de contrôle pour mon cholestérol",
    "Petite éruption cutanée au bras, pas de démangeaison, pas de fièvre",
]

# Justifications par niveau
RAISONS = {
    "maximale": [
        "Urgence vitale, transport médicalisé prioritaire",
        "Risque d'AVC, imagerie cérébrale urgente",
        "Détresse respiratoire, surveillance continue",
        "Hémorragie active, hémostase et transfusion",
        "Urgence chirurgicale, bloc opératoire",
        "Risque anaphylactique,adrénaline et surveillance",
        "Convulsions fébriles pédiatriques, hospitalisation",
        "Traumatisme médullaire suspecté, immobilisation",
        "IDM suspecté, ECG et thrombolyse",
        "Brûlure grave, centre spécialisé grands brûlés",
    ],
    "modérée": [
        "Examen clinique complémentaire dans les 4h",
        "Radiographie et bilan sanguin",
        "Surveillance et antalgiques, réévaluation à 6h",
        "Antibiothérapie ciblée, contrôle à 24h",
        "Suture ou pansement spécialisé",
        "Déshydratation, perfusion et rééquilibration",
        "Consultation ORL ou orthopédique rapide",
        "Antalgiques palier 2 et surveillance",
        "Plaie suturée, suivi à 48h",
        "Antihistaminique et contrôle allergologue",
    ],
    "différée": [
        "Programmer une consultation de médecine générale",
        "Renouvellement d'ordonnance, médecin traitant",
        "Conseils hygiéno-diététiques et surveillance",
        "Bilan de routine, sans caractère d'urgence",
        "Consultation spécialisée sous 4 semaines",
        "Symptômes chroniques stables, suivi habituel",
        "Auto-soins et réévaluation si aggravation",
        "Avis spécialisé non urgent",
        "Kiné et rééducation fonctionnelle",
        "Éducation thérapeutique et observance",
    ],
}


def generate_dataset(size: int, seed: int = 42) -> list[dict]:
    """Produit `size` exemples synthétiques équilibrés."""
    rng = random.Random(seed)
    bucket = {
        "maximale": MAXIMALE_PHRASES,
        "modérée": MODEREE_PHRASES,
        "différée": DIFFEREE_PHRASES,
    }

    # Distribution : 30% maximale, 40% modérée, 30% différée (réaliste côté urgences)
    distribution = (
        ["maximale"] * int(size * 0.30)
        + ["modérée"] * int(size * 0.40)
        + ["différée"] * (size - int(size * 0.30) - int(size * 0.40))
    )
    rng.shuffle(distribution)

    items = []
    for niveau in distribution:
        text = rng.choice(bucket[niveau])
        # 5% de bruit : niveau volontairement ambigu, à 0.4 de confiance
        if rng.random() < 0.05:
            labeled_niveau = niveau
            confidence = round(rng.uniform(0.35, 0.55), 2)
        else:
            labeled_niveau = niveau
            confidence = round(rng.uniform(0.7, 0.98), 2)
        items.append(
            {
                "text": text,
                "niveau": labeled_niveau,
                "raison": rng.choice(RAISONS[labeled_niveau]),
                "llm_confidence": confidence,
            }
        )
    return items


def main() -> None:
    p = argparse.ArgumentParser(description="Génère un dataset de triage synthétique.")
    p.add_argument("--output", default="data/processed/labeled_triage.jsonl")
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    items = generate_dataset(args.size, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    by_label = {}
    for item in items:
        by_label[item["niveau"]] = by_label.get(item["niveau"], 0) + 1

    print(f"✅ {len(items)} exemples générés dans {output}")
    print(f"📊 Distribution : {by_label}")


if __name__ == "__main__":
    main()
