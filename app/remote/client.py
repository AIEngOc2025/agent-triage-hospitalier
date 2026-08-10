import json
import logging
import os
from typing import AsyncGenerator, List, Literal, Optional

import httpx
import instructor
from openai import AsyncOpenAI

from app.schemas import TriageResponse
from app.timing import time_execution

logger = logging.getLogger(__name__)
InferenceMode = Literal["conversationnel", "structured"]


class RemoteInferenceClient:
    def __init__(
        self,
        inference_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 60,  # Réduit à 60 : suffisant pour un JSON de triage
        repetition_penalty: float = 1.1,
        timeout: float = 30.0,  # 30s est suffisant si le modèle ne "délire" plus
    ):
        self.inference_url = inference_url or os.getenv(
            "INFERENCE_SERVICE_URL",
            "https://agent-inference-service-414294705487.europe-west1.run.app",
        )
        self.model_name = model_name or os.getenv(
            "MODEL_PATH", "/app/models/merged_dpo_final_chsa"
        )

        # Pool de connexions HTTP permanent
        self.raw_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )

        self.openai = AsyncOpenAI(
            base_url=f"{self.inference_url}/v1",
            api_key="EMPTY",
            http_client=self.raw_client,
        )

        # OPTIMISATION : Mode JSON pur (plus rapide que MD_JSON sur vLLM)
        self.structured_client = instructor.from_openai(
            self.openai, mode=instructor.Mode.JSON
        )

        self.params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
        }

    def _prepare_payload(self, messages: List[dict], stream: bool) -> dict:
        """Prépare le payload avec des séquences d'arrêt strictes."""
        return {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            **self.params,
            # Arrête le modèle dès qu'il finit le JSON ou change de ligne
            "stop": ["\n", "}", "###"],
            "extra_body": {
                # Force le format JSON au niveau du moteur vLLM (optimisation Outlines)
                "guided_json": TriageResponse.model_json_schema()
            },
        }

    @time_execution("network_inference")
    async def generate(self, messages: List[dict]) -> str:
        payload = self._prepare_payload(messages, stream=False)
        response = await self.raw_client.post(
            f"{self.inference_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @time_execution("network_inference_structured")
    async def generate_structured(self, messages: List[dict]) -> TriageResponse:
        """Génération structurée sans retries pour minimiser la latence."""
        try:
            return await self.structured_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_model=TriageResponse,
                # On met max_retries à 0 pour le benchmark afin de voir
                # la vitesse réelle.
                # En production, on peut mettre 1.
                max_retries=0,
                temperature=0,
                extra_body={"guided_json": TriageResponse.model_json_schema()},
            )
        except Exception as e:
            logger.error(f"❌ Structured inference failed: {e}")
            raise

    async def generate_stream(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        payload = self._prepare_payload(messages, stream=True)
        async with self.raw_client.stream(
            "POST", f"{self.inference_url}/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    content = line[6:].strip()
                    if content == "[DONE]":
                        break
                    try:
                        chunk = json.loads(content)
                        if delta := chunk["choices"][0]["delta"].get("content"):
                            yield delta
                    except json.JSONDecodeError:
                        continue

    async def close(self) -> None:
        await self.raw_client.aclose()
