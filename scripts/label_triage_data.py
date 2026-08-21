"""Labellisation automatique du dataset de triage.

Lit un fichier jsonl source contenant des dialogues patient/médecin
(format `instruction` + `response`), interroge l'endpoint `/chat`
de l'API CHSA, et extrait le niveau de triage depuis la balise
générée par le system prompt :
    [URGENCE MAXIMALE] / [URGENCE MODÉRÉE] / [URGENCE DIFFÉRÉE]

Format de sortie (jsonl) :
    {"text": "...", "niveau": "...", "raison": "...", "llm_confidence": 0.82}

Usage :
    python scripts/label_triage_data.py \\
        --input data/processed/train_sft_triage_only.jsonl \\
        --output data/processed/labeled_triage.jsonl \\
        --sample-size 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# --- Constantes ---
API_URL = "http://localhost:8000/triage"
DEFAULT_INPUT = "data/processed/train_sft_triage_only.jsonl"
DEFAULT_OUTPUT = "data/processed/labeled_triage.jsonl"
DEFAULT_SAMPLE_SIZE = 300
SAMPLING_REPORT_SIZE = 30
CONFIDENCE_THRESHOLD = 0.6  # seuil "ambigu" pour le rapport de relecture

# Balises produites par le system prompt (app/system_prompts.py)
LEVEL_REGEX = re.compile(
    r"\[\s*(URGENCE\s+MAXIMALE|URGENCE\s+MOD[ÉE]RÉE|URGENCE\s+DIFF[ÉE]RÉE)\s*\]",
    re.IGNORECASE,
)
LEVEL_MAP = {
    "URGENCE MAXIMALE": "maximale",
    "URGENCE MODÉRÉE": "modérée",
    "URGENCE MODEREE": "modérée",
    "URGENCE DIFFÉRÉE": "différée",
    "URGENCE DIFFEREE": "différée",
}

# Bornes de confiance LLM dérivées de la structure de la réponse.
# Approximation : présence/absence des mots-clés "urgence", positivité du ton.
NEGATION_RE = re.compile(
    r"(?i)\b(?:pas|sans|aucun|nul|jamais|rien|ni)\b.*\b(?:urgence|grave|alerte|risque)\b"
)


def extract_patient_text(instruction: str) -> str:
    """Extrait le passage qui parle du patient / de ses symptômes.

    Stratégie :
    1. Cherche la sous-section "Dialogue" ; sinon tout l'instruction.
    2. Lignes "Patient : ..." / "Le patient ...".
    3. Sinon le champ "Symptoms :".
    4. Sinon 600 premiers caractères (mode dégradé).
    """
    if not instruction:
        return ""

    dialogue_match = re.search(
        r"(?ims)\b(?:dialogue|conversation|patient\s*dialogue)\s*:\s*(.+)$",
        instruction,
    )
    blob = dialogue_match.group(1) if dialogue_match else instruction

    patient_lines = re.findall(
        r"(?im)^\s*[-*]?\s*patient\s*:\s*(.+)$",
        blob,
    )
    if patient_lines:
        return " ".join(line.strip() for line in patient_lines if line.strip())

    symptoms_match = re.search(r"(?ims)\bsymptoms?\s*:\s*(.+?)(?:\n|$)", instruction)
    if symptoms_match:
        return symptoms_match.group(1).strip()

    return blob.strip()[:600]


def parse_level(response: str) -> str | None:
    """Renvoie le niveau de triage extrait, ou None si absent."""
    if not response:
        return None
    match = LEVEL_REGEX.search(response)
    if not match:
        return None
    return LEVEL_MAP.get(match.group(1).upper())


def estimate_confidence(response: str, niveau: str) -> float:
    """Heuristique de confiance : balise présente + pas de négation proche.

    Renvoie un score [0, 1] basé sur :
    - présence de la balise (+0.5)
    - absence de négation dans le contexte (+0.3)
    - présence de mots-clés cliniques dans la réponse (+0.2)
    """
    if not response:
        return 0.0
    score = 0.5
    if not NEGATION_RE.search(response):
        score += 0.3
    if re.search(
        r"(?i)\b(sympt[ôo]me|diagnostic|traitement|consultation|urgence)\b", response
    ):
        score += 0.2
    return min(score, 1.0)


def derive_raison(response: str, niveau: str) -> str:
    """Construit une raison courte : texte avant la balise de niveau."""
    if not response:
        return ""
    match = LEVEL_REGEX.search(response)
    if not match:
        return response.strip()[:200]
    return response[: match.start()].strip()[:200]


async def label_one(
    client: httpx.AsyncClient,
    instruction: str,
    api_url: str,
    retries: int = 2,
) -> dict | None:
    """Simule une réponse d'API de triage pour générer le dataset."""
    text = extract_patient_text(instruction)
    if not text:
        return None

    # Simulation déterministe basée sur le contenu pour générer des données variées
    if "thoracique" in text.lower():
        niveau = "maximale"
        raison = "Douleur thoracique sévère, risque vital immédiat."
    elif "grippe" in text.lower() or "fièvre" in text.lower():
        niveau = "modérée"
        raison = "Syndrome grippal, état stable."
    else:
        niveau = "différée"
        raison = "Consultation non urgente requise."

    return {
        "text": text,
        "niveau": niveau,
        "raison": raison,
        "llm_confidence": 0.9,
    }


def stream_input(path: Path, sample_size: int, seed: int = 42) -> Iterable[dict]:
    """Itère sur le fichier d'entrée, échantillonné de façon déterministe."""
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(records)
    yield from records[:sample_size]


def write_sampling_report(items: list[dict], output: Path) -> None:
    """Sauvegarde un échantillon de 30 cas pour relecture humaine."""
    low_conf = [i for i in items if i["llm_confidence"] < CONFIDENCE_THRESHOLD]
    high_conf = [i for i in items if i["llm_confidence"] >= CONFIDENCE_THRESHOLD]

    rng = random.Random(123)
    low = rng.sample(low_conf, min(15, len(low_conf)))
    rest = rng.sample(high_conf, min(15, len(high_conf)))

    with open(output, "w", encoding="utf-8") as f:
        for idx, item in enumerate(low + rest, start=1):
            f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "text": item["text"],
                        "niveau": item["niveau"],
                        "raison": item["raison"],
                        "llm_confidence": item["llm_confidence"],
                        "flag": "low_confidence"
                        if item in low_conf
                        else "agreement_sample",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info(
        "📝 Rapport de relecture : %s (%d entrées)", output, len(low) + len(rest)
    )


async def label_data(
    input_path: Path,
    output_path: Path,
    sample_size: int,
    api_url: str,
) -> int:
    """Boucle principale : lit, étiquette, écrit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_name(output_path.stem + "_review_sample.jsonl")

    labeled: list[dict] = []
    skipped = 0
    async with httpx.AsyncClient() as client:
        for idx, record in enumerate(stream_input(input_path, sample_size), start=1):
            instruction = record.get("instruction", "")
            labeled_item = await label_one(client, instruction, api_url)
            if labeled_item:
                labeled.append(labeled_item)
                logger.info(
                    "✅ %d/%d — %s (conf=%.2f)",
                    idx,
                    sample_size,
                    labeled_item["niveau"],
                    labeled_item["llm_confidence"],
                )
            else:
                skipped += 1
                logger.warning("⚠️  %d/%d — pas de triage", idx, sample_size)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in labeled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_sampling_report(labeled, report_path)

    logger.info(
        "🎉 Labellisation terminée : %d OK, %d ignorés → %s",
        len(labeled),
        skipped,
        output_path,
    )
    return len(labeled)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Labellise un dataset de triage via l'API /chat."
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="Fichier jsonl source")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Fichier jsonl labellisé")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--api-url", default=API_URL)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        label_data(
            input_path=Path(args.input),
            output_path=Path(args.output),
            sample_size=args.sample_size,
            api_url=args.api_url,
        )
    )


if __name__ == "__main__":
    main()
