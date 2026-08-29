from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NiveauTriage(str, Enum):
    """Niveaux de triage reconnus par le CHSA.

    Aligné avec le system prompt `app/system_prompts.py` et le
    `guided_regex` côté vLLM (`app/remote/client.py`).
    """

    MAXIMALE = "maximale"
    MODEREE = "modérée"
    DIFFEREE = "différée"


class TriageResult(BaseModel):
    """Résultat structuré du triage, retourné par le LLM lorsque la
    situation est suffisamment claire pour classifier.

    Champs :
        niveau : niveau de triage, ou None si la conversation se poursuit
        orientation : explication courte de l'orientation recommandée
    """

    niveau: Optional[NiveauTriage] = Field(
        None,
        description=(
            "Niveau de triage : maximale, modérée, ou différée. "
            "Reste null tant que la conversation n'a pas permis de classifier."
        ),
    )
    orientation: Optional[str] = Field(
        None,
        description="Explication courte de l'orientation recommandée si connue.",
    )


class TriageResponse(BaseModel):
    """Structure de réponse du modèle : un message conversationnel
    bienveillant obligatoire, et un triage optionnel.

    Cette structure est consommée par `instructor.from_openai` côté
    `app/remote/client.py` (cf. §3.2.7 du rapport technique).
    """

    message: str = Field(
        ...,
        description=(
            "Réponse conversationnelle humaine, bienveillante et concise "
            "destinée au patient. Toujours présente."
        ),
    )
    triage_result: Optional[TriageResult] = Field(
        None,
        description=(
            "Résultat final de triage. À remplir UNIQUEMENT si la situation "
            "est claire et la classification possible."
        ),
    )
