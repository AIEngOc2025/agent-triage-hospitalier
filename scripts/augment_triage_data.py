"""Augmentation locale d'un dataset de triage par règles.

Génère des paraphrases contrôlées (substitutions de synonymes,
variations de format, etc.) sans dépendre d'un LLM.

Pourquoi local ?
- Le LLM CHSA est biaisé vers le haut (surclasse modérée en maximale)
- Les labels NLIE2 / new_triage sont issus de grilles START/JumpSTART
  -> on les garde comme vérité terrain
- L'augmentation par règles est déterministe et ne dégrade pas la qualité des labels

Usage :
    python scripts/augment_triage_data.py \\
        --input data/processed/consolidated_labeled.jsonl \\
        --output data/processed/augmented_labeled.jsonl \\
        --target-size 1000
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# --- Dictionnaires de synonymes (médical, FR/EN) ---

# Dictionnaire de traduction EN→FR pour termes médicaux courants
EN_TO_FR = {
    "pain": "douleur",
    "chest pain": "douleur thoracique",
    "abdominal pain": "douleur abdominale",
    "headache": "maux de tête",
    "severe headache": "céphalées sévères",
    "fever": "fièvre",
    "high fever": "fièvre élevée",
    "temperature": "température",
    "vomiting": "vomissements",
    "nausea": "nausées",
    "shortness of breath": "essoufflement",
    "difficulty breathing": "difficulté respiratoire",
    "rash": "éruption cutanée",
    "skin rash": "éruption",
    "eruption": "éruption",
    "dizziness": "vertiges",
    "vertigo": "vertige",
    "fatigue": "fatigue",
    "tiredness": "fatigue",
    "bleeding": "saignement",
    "hemorrhage": "hémorragie",
    "cough": "toux",
    "coughing": "toux",
    "diarrhea": "diarrhée",
    "constipation": "constipation",
    "weakness": "faiblesse",
    "numbness": "engourdissement",
    "sweating": "transpiration",
    "cold sweats": "sueurs froides",
    "palpitations": "palpitations",
    "rapid heartbeat": "battements cardiaques rapides",
    "confusion": "confusion",
    "unconscious": "inconscient",
    "unresponsive": "sans réponse",
    "seizure": "convulsion",
    "convulsions": "convulsions",
    "trauma": "traumatisme",
    "injury": "blessure",
    "wound": "plaie",
    "fracture": "fracture",
    "broken": "cassée",
    "allergic reaction": "réaction allergique",
    "anaphylaxis": "anaphylaxie",
    "difficulty swallowing": "difficulté à avaler",
    "blood pressure": "tension artérielle",
    "hypertension": "hypertension",
    "year-old": "ans",
    "y/o": "ans",
    "yo": "ans",
    "male": "homme",
    "female": "femme",
    "presents with": "présente",
    "complains of": "se plaint de",
    "the patient": "le patient",
    "patient": "patient",
    "abdominal": "abdominale",
    "severe": "sévère",
    "acute": "aiguë",
    "chronic": "chronique",
    "since": "depuis",
    "for": "depuis",
    "days": "jours",
    "hours": "heures",
    "with": "avec",
    "and": "et",
    "of": "de",
}

# Synonymes de mots-clés cliniques (substitution intra-langue)
SYNONYMS = {
    # FR — Douleur
    "douleur": ["souffrance", "mal", "gêne douloureuse", "sensation douloureuse"],
    "douleur thoracique": [
        "douleur au thorax",
        "douleur à la poitrine",
        "serrement thoracique",
    ],
    "douleur abdominale": ["mal au ventre", "douleur au bas-ventre", "gêne abdominale"],
    # FR — Fièvre
    "fièvre": ["température", "fébrilité", "hyperthermie"],
    "fièvre élevée": ["forte température", "fièvre importante", "température à 39"],
    # FR — Vomissements
    "vomissements": ["vomissures", "émèses", "nausées avec rejet"],
    # FR — Maux de tête
    "maux de tête": ["céphalées", "douleurs crâniennes"],
    "céphalées": ["maux de tête", "douleurs crâniennes"],
    # FR — Essoufflement
    "essoufflement": ["dyspnée", "gène respiratoire", "difficulté à respirer"],
    "dyspnée": ["essoufflement", "gène respiratoire"],
    # FR — Fatigue
    "fatigue": ["asthénie", "épuisement", "lassitude"],
    # FR — Éruption
    "éruption": ["rash", "érythème", "plaques cutanées"],
    "éruption cutanée": ["rash", "plaques", "rougeurs cutanées"],
    # FR — Vertige
    "vertige": ["étourdissement", "sensation de tête qui tourne"],
    "vertiges": ["étourdissements", "sensations vertigineuses"],
    # FR — Saignement
    "saignement": ["hémorragie", "écoulement sanguin"],
    # FR — Crampes
    "crampes": ["contractions musculaires", "spasmes"],
    # FR — Toux
    "toux": ["toux persistante", "expectorations"],
    # FR — Patient
    "patient_fr": ["sujet", "personne", "individu"],
    # FR — Sueurs
    "sueurs froides": ["transpiration abondante", "diaphorèse"],
    "transpiration": ["sueurs", "diaphorèse"],
    # EN — Douleur
    "pain": ["discomfort", "tenderness", "soreness", "aching"],
    "chest pain": ["chest discomfort", "pressure in chest", "chest tightness"],
    "abdominal pain": ["belly pain", "stomach pain", "abdominal discomfort"],
    # EN — Headache
    "headache": ["cephalalgia", "head pain"],
    "severe headache": ["intense headache", "splitting headache"],
    # EN — Fever
    "fever": ["temperature", "febrile state", "pyrexia"],
    # EN — Vomiting
    "vomiting": ["emesis", "throwing up", "retching"],
    # EN — Shortness of breath
    "shortness of breath": ["dyspnea", "labored breathing", "breathing difficulty"],
    # EN — Rash
    "rash": ["eruption", "skin outbreak", "exanthem", "skin rash"],
    # EN — Dizziness
    "dizziness": ["vertigo", "lightheadedness", "spinning sensation"],
    # EN — Bleeding
    "bleeding": ["hemorrhage", "blood loss"],
    # EN — Patient
    "patient_en": ["subject", "individual", "person"],
    # year-old
    "year-old": ["y/o", "-year-old", "year old"],
}

# Templates d'enveloppe pour ajouter de la variation d'introduction
INTRODUCTION_TEMPLATES_FR = [
    "Patient avec les symptômes suivants : {symptoms}",
    "Symptômes rapportés : {symptoms}",
    "Motif de consultation : {symptoms}",
    "Le patient se plaint de : {symptoms}",
    "À l'admission : {symptoms}",
    "Tableau clinique : {symptoms}",
    "Histoire de la maladie : {symptoms}",
    "Anamnèse : {symptoms}",
    "À l'examen : {symptoms}",
    "Plaintes : {symptoms}",
    "Le patient décrit : {symptoms}",
    "Un patient de {age} ans, {symptoms}",
    "Une patiente de {age} ans, {symptoms}",
    "Un homme de {age} ans, {symptoms}",
    "Une femme de {age} ans, {symptoms}",
    "Ce patient de {age} ans, {symptoms}",
    "À l'interrogatoire, le patient signale : {symptoms}",
    "Le motif de venue est : {symptoms}",
    "Le patient rapporte : {symptoms}",
    "Le patient consulte pour : {symptoms}",
    "Devant l'apparition de : {symptoms}",
    "Dans un contexte de : {symptoms}",
    "Survenue de : {symptoms}",
    "Patient adressé pour : {symptoms}",
    "Cas clinique : {symptoms}",
    "Dossier : patient avec {symptoms}",
    "Le SAMU amène un patient avec {symptoms}",
]

INTRODUCTION_TEMPLATES_EN = [
    "Patient presents with {symptoms}",
    "Chief complaint: {symptoms}",
    "History of present illness: {symptoms}",
    "The patient reports {symptoms}",
    "On examination: {symptoms}",
    "Clinical features: {symptoms}",
    "A {age} presents with {symptoms}",
    "A {age} male with {symptoms}",
    "A {age} female with {symptoms}",
    "The patient is a {age} with {symptoms}",
]

# Variantes de formats d'âge
AGE_PATTERNS = [
    (r"(\d+)-year-old female", r"\1 y/o female"),
    (r"(\d+)-year-old male", r"\1 y/o male"),
    (r"(\d+)-year-old", r"\1-year-old"),
    (r"(\d+) y/o", r"\1-year-old"),
    (r"(\d+) yo", r"\1-year-old"),
]

# Variations de ponctuation
PUNCTUATION_VARIANTS = [
    (",", ", "),
    (",", ";"),
    (";", ","),
    (" and ", " & "),
    (" and ", " + "),
    (" et ", " & "),
]


def substitute_synonyms(text: str, rng: random.Random) -> str:
    """Remplace ~30% des synonymes trouvés."""

    def replace(match):
        word = match.group(0).lower()
        if rng.random() < 0.3 and word in SYNONYMS:
            return rng.choice(SYNONYMS[word])
        return match.group(0)

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in SYNONYMS) + r")\b", re.IGNORECASE
    )
    return pattern.sub(replace, text)


def translate_en_to_fr(text: str, rng: random.Random) -> str:
    """Traduit un texte anglais en français par substitutions simples.

    Limitation : ne couvre que les termes présents dans EN_TO_FR.
    Pour les termes inconnus, on garde le texte original (souvent médical).
    """
    result = text
    # Tri par longueur décroissante pour matcher les expressions longues en premier
    keys = sorted(EN_TO_FR.keys(), key=lambda x: -len(x))
    n_translated = 0
    for en_term in keys:
        fr_term = EN_TO_FR[en_term]
        pattern = re.compile(r"\b" + re.escape(en_term) + r"\b", re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub(fr_term, result)
            n_translated += 1
    # Garde seulement si au moins 2 termes ont été traduits
    if n_translated < 2:
        return text
    return result


def apply_age_variant(text: str, rng: random.Random) -> str:
    """Applique une variation de format d'âge."""
    for pat, repl in AGE_PATTERNS:
        if re.search(pat, text) and rng.random() < 0.4:
            new_text = re.sub(pat, repl, text, count=1)
            # Évite les artefacts (double-tiret, etc.)
            if "--" not in new_text:
                text = new_text
                break
    return text


def apply_punctuation_variant(text: str, rng: random.Random) -> str:
    """Applique une variation de ponctuation."""
    for old, new in PUNCTUATION_VARIANTS:
        if old in text and rng.random() < 0.3:
            text = text.replace(old, new, 1)
            break
    return text


def wrap_introduction(text: str, rng: random.Random) -> str:
    """Enveloppe les symptômes (courts) dans un template d'introduction.

    Stratégie : n'enveloppe QUE si le texte contient une séquence de
    symptômes typique (virgules, "with", "presents with"). Sinon,
    garder le texte brut évite la concaténation artefactuelle.
    """
    # Si le texte est déjà narratif (contient "presents with", "with", etc.)
    # on ne tente pas d'enveloppe, on le laisse tel quel
    if re.search(
        r"\bpresents with\b|\bChief complaint\b|\bHistory\b|\bpatient reports\b",
        text,
        re.IGNORECASE,
    ):
        return text

    # Détermine la langue
    is_fr = bool(re.search(r"[éèêàù]", text)) or any(
        w in text.lower() for w in ["le patient", "la patiente", "douleur", "fièvre"]
    )
    templates = INTRODUCTION_TEMPLATES_FR if is_fr else INTRODUCTION_TEMPLATES_EN

    # Coupe court : si > 200 chars, l'enveloppe dégradait le texte
    if len(text) > 200:
        return text

    # Pour des symptômes courts (liste séparée par virgules), enveloppe
    if "," in text and len(text) < 100:
        template = rng.choice(templates)
        return template.format(symptoms=text, age="32")

    return text


def augment_record(record: dict, rng: random.Random) -> list[dict]:
    """Génère 1-3 paraphrases d'un exemple."""
    augmented = []
    text = record["text"]
    niveau = record["niveau"]

    # Stratégie 1 : synonymes
    v1 = substitute_synonyms(text, rng)
    v1 = clean_artefacts(v1)
    if v1 != text and v1 not in (a["text"] for a in augmented):
        augmented.append({"text": v1, "niveau": niveau})

    # Stratégie 2 : variation d'âge + ponctuation
    v2 = apply_age_variant(text, rng)
    v2 = apply_punctuation_variant(v2, rng)
    v2 = clean_artefacts(v2)
    if v2 != text and v2 not in (a["text"] for a in augmented):
        augmented.append({"text": v2, "niveau": niveau})

    # Stratégie 3 : enveloppe d'introduction (surtout si texte court)
    if len(text) < 150:
        v3 = wrap_introduction(text, rng)
        v3 = clean_artefacts(v3)
        if v3 != text and v3 not in (a["text"] for a in augmented):
            augmented.append({"text": v3, "niveau": niveau})

    # Stratégie 4 : traduction EN→FR (si applicable)
    if detect_lang(text) == "en":
        v4 = translate_en_to_fr(text, rng)
        v4 = clean_artefacts(v4)
        if (
            v4 != text
            and v4 not in (a["text"] for a in augmented)
            and detect_lang(v4) == "fr"
        ):
            augmented.append({"text": v4, "niveau": niveau})

    return augmented


def detect_lang(text: str) -> str:
    """Détection linguistique simple."""
    if any(c in text for c in "éèêàùçôî") or any(
        w in text.lower()
        for w in [
            "le patient",
            "douleur",
            "fièvre",
            "depuis",
            "jours",
            "patient",
            "présente",
            "vomissements",
        ]
    ):
        return "fr"
    if any(
        w in text.lower()
        for w in [
            "the patient",
            "presents",
            "with",
            "year-old",
            "history",
            "chief complaint",
            "shortness of breath",
            "chest pain",
            "headache",
            "nausea",
            "vomiting",
            "pain",
        ]
    ):
        return "en"
    return "unknown"


def clean_artefacts(text: str) -> str:
    """Nettoie artefacts résiduels (double-tirets, espaces multiples)."""
    text = re.sub(r"--+", "-", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def augment_dataset(
    records: list[dict],
    target_size: int,
    seed: int = 42,
    target_fr_ratio: float = 0.40,
) -> list[dict]:
    """Augmente jusqu'à target_size par paraphrase.

    Garantit un ratio cible de français (target_fr_ratio par défaut 40%).
    """
    rng = random.Random(seed)
    output = list(records)

    def current_fr_ratio(out):
        if not out:
            return 0.0
        n_fr = sum(1 for r in out if detect_lang(r["text"]) == "fr")
        return n_fr / len(out)

    while len(output) < target_size:
        # Décide de la stratégie : traduire ou paraphraser
        cur_fr = current_fr_ratio(output)
        want_fr = cur_fr < target_fr_ratio

        # Tirage équilibrant les classes sous-représentées
        dist = {}
        for r in output:
            dist[r["niveau"]] = dist.get(r["niveau"], 0) + 1

        weights = {k: 1.0 / max(1, v) for k, v in dist.items()}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        niveaux = list(weights.keys())
        weights_list = [weights[n] for n in niveaux]
        chosen_niveau = rng.choices(niveaux, weights=weights_list, k=1)[0]

        # Tire un record selon stratégie (FR ou autre)
        candidates_all = [r for r in records if r["niveau"] == chosen_niveau]
        if not candidates_all:
            continue
        if want_fr:
            en_candidates = [
                r for r in candidates_all if detect_lang(r["text"]) == "en"
            ]
            candidates = en_candidates if en_candidates else candidates_all
        else:
            candidates = candidates_all

        record = rng.choice(candidates)

        # Génère paraphrases (jusqu'à 2 par appel)
        augmented = augment_record(record, rng)
        if augmented:
            # Si on veut du FR, on privilégie les augmentations FR
            if want_fr:
                fr_augs = [a for a in augmented if detect_lang(a["text"]) == "fr"]
                if fr_augs:
                    output.extend(fr_augs[:1])
                    continue
            output.extend(augmented[:1])

    return output[:target_size]


def main() -> None:
    p = argparse.ArgumentParser(description="Augmente un dataset de triage par règles.")
    p.add_argument("--input", default="data/processed/consolidated_labeled.jsonl")
    p.add_argument("--output", default="data/processed/augmented_labeled.jsonl")
    p.add_argument("--target-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    records = [json.loads(line) for line in open(args.input) if line.strip()]
    print(f"📂 Input : {len(records)} cas")

    augmented = augment_dataset(records, args.target_size, args.seed)

    dist = {}
    for r in augmented:
        dist[r["niveau"]] = dist.get(r["niveau"], 0) + 1
    print(f"📊 Output : {len(augmented)} cas | Distribution : {dist}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in augmented:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"💾 Saved → {args.output}")


if __name__ == "__main__":
    main()
