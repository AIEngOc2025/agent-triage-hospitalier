"""Client d'inférence distant — supporte deux modes :

- **structured** : utilise `instructor.from_openai` pour obtenir une
  réponse Pydantic typée (`TriageResponse`). Validation stricte + retry
  automatique sur erreur de validation.
- **conversationnel** : utilise `httpx` brut pour générer du texte
  libre (chat conversationnel sans classification).

L'architecture conversationnelle privilégie le mode conversationnel avec
`guided_regex` vLLM comme garde-fou. Le mode structuré est utilisé en
complément pour les routes `/triage` (Prio 3 du plan d'intégration).
"""

import json
import logging
import os
from typing import AsyncGenerator, List, Literal, Optional

import httpx
from openai import AsyncOpenAI

import instructor

from app.schemas import TriageResponse
from app.timing import time_execution

logger = logging.getLogger(__name__)

InferenceMode = Literal["conversationnel", "structured"]


class RemoteInferenceClient:
    """Client polymorphe pour interagir avec un service d'inference
    compatible OpenAI (vLLM).
    """

    def __init__(
        self,
        inference_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        timeout: float = 300.0,
        mode: InferenceMode = "conversationnel",
    ):
        self.inference_url = inference_url or os.getenv(
            "INFERENCE_SERVICE_URL",
            "https://agent-inference-service-414294705487.europe-west1.run.app",
        )
        self.model_name = model_name or os.getenv(
            "MODEL_PATH", "/app/models/merged_dpo_final_chsa"
        )
        self.temperature = float(temperature or os.getenv("TEMPERATURE", 0.1))
        self.max_tokens = int(max_tokens or os.getenv("MAX_TOKENS", 50))
        self.repetition_penalty = float(
            repetition_penalty or os.getenv("REPETITION_PENALTY", 1.5)
        )
        self.timeout = timeout
        self.mode = mode

        # Client HTTP brut (mode conversationnel)
        self.raw_client = httpx.AsyncClient(timeout=timeout)

        # Client OpenAI + instructor (mode structured)
        # vLLM accepte n'importe quelle clé API key, on force "EMPTY".
        self.openai = AsyncOpenAI(
            base_url=f"{self.inference_url}/v1",
            api_key="EMPTY",
            timeout=timeout,
            max_retries=2,
        )
        self.structured_client = instructor.from_openai(
            self.openai, mode=instructor.Mode.MD_JSON
        )

        logger.info(
            f"✅ [REMOTE] Client initialized at {self.inference_url} "
            f"with model {self.model_name}. Mode={self.mode}. "
            f"Params: temp={self.temperature}, "
            f"max_tokens={self.max_tokens}, "
            f"penalty={self.repetition_penalty}"
        )

    def _prepare_payload(self, messages: List[dict], stream: bool) -> dict:
        """Prépare le payload pour un appel conversationnel (mode legacy).

        Le `guided_regex` force le format `\[Niveau: ...\] - Orientation : ...`
        pour permettre la rétrocompatibilité avec les versions sans
        `instructor`. En mode `structured`, le rendu structuré
        (`TriageResponse`) prime et le format est validé côté client.
        """
        # NOTE: le format "triage_result" est désormais utilisé en sortie
        # structurée (cf. `system_prompts.py`). Le guided_regex est conservé
        # comme garde-fou de compatibilité.
        regex_pattern = r"\{.*\"triage_result\".*\}"
        return {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "extra_body": {"guided_regex": regex_pattern},
        }

    @time_execution("network_inference")
    async def generate(self, messages: List[dict]) -> str:
        """Génère une réponse conversationnelle (texte brut).

        Utilise le mode `conversationnel` via httpx. Le `guided_regex`
        force le format côté vLLM. La réponse est nettoyée en amont par
        `clean_response` côté API Gateway (`app/main.py`).
        """
        payload = self._prepare_payload(messages, stream=False)
        logger.debug(
            f"Calling {self.inference_url}/v1/chat/completions with "
            f"payload: {json.dumps(payload)}"
        )
        try:
            response = await self.raw_client.post(
                f"{self.inference_url}/v1/chat/completions",
                json=payload,
            )
            if response.status_code != 200:
                logger.error(f"Error response body: {response.text}")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            logger.error(f"❌ HTTP Error during generation: {e}")
            raise

    @time_execution("network_inference_structured")
    async def generate_structured(
        self, messages: List[dict]
    ) -> TriageResponse:
        """Génère une réponse structurée typée `TriageResponse` via
        `instructor`.

        Avantages :
        - Validation Pydantic stricte côté client
        - Retry automatique (max_retries=2) sur erreur de validation
        - Mode MD_JSON : sérialisation via Markdown JSON, robuste

        Returns:
            TriageResponse validé.
        """
        try:
            response = await self.structured_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_model=TriageResponse,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_retries=2,
            )
            return response
        except Exception as e:
            logger.error(f"❌ Structured inference failed: {e}")
            raise

    @time_execution("network_inference_stream")
    async def generate_stream(
        self, messages: List[dict]
    ) -> AsyncGenerator[str, None]:
        """Génère une réponse en streaming (texte brut, mode conversationnel)."""
        payload = self._prepare_payload(messages, stream=True)
        logger.debug(
            f"Calling {self.inference_url}/v1/chat/completions with "
            f"streaming payload: {json.dumps(payload)}"
        )
        try:
            async with self.raw_client.stream(
                "POST",
                f"{self.inference_url}/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        content = line[len("data:") :].strip()
                        if content != "[DONE]":
                            try:
                                chunk = json.loads(content)
                                if "content" in chunk["choices"][0]["delta"]:
                                    yield chunk["choices"][0]["delta"]["content"]
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Could not decode JSON from stream line: {content}"
                                )
        except httpx.HTTPError as e:
            logger.error(f"❌ HTTP Error during streaming generation: {e}")
            raise

    async def close(self) -> None:
        """Ferme les deux clients HTTP proprement."""
        await self.raw_client.aclose()
        await self.openai.close()
        logger.info("🔌 [REMOTE] Both clients closed successfully.")
