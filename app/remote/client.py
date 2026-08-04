import json
import os
from typing import AsyncGenerator, List

import httpx


class RemoteInferenceClient:
    """
    Client for interacting with a remote inference service.
    """

    def __init__(self):
        """
        @definition : Initialise le client d'inférence distante avec
        ses configurations et le client HTTP.
        @args/params : Aucun
        @return : Aucun
        """
        self.inference_url = os.getenv(
            "INFERENCE_SERVICE_URL",
            "https://agent-inference-service-rlgcjqsysq-ew.a.run.app",
        )
        self.model_name = os.getenv("MODEL_PATH", "/app/models/merged_dpo_final_chsa")
        self.client = httpx.AsyncClient(timeout=300.0)
        print(
            f"✅ [REMOTE] Remote Inference Client initialized at "
            f"{self.inference_url} with model {self.model_name}"
        )

    async def generate(self, messages: List[dict]) -> str:
        """
        @definition : Génère une réponse complète (non-streaming)
        depuis le service d'inférence.
        @args/params : messages (List[dict]): L'historique des
        messages de la conversation.
        @return : str : Le contenu textuel de la réponse générée.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "temperature": 0.05,
            "max_tokens": 150,
            "repetition_penalty": 1.5,
        }
        print(
            f"DEBUG: Calling {self.inference_url}/v1/chat/completions "
            f"with payload: {json.dumps(payload)}"
        )

        try:
            response = await self.client.post(
                f"{self.inference_url}/v1/chat/completions",
                json=payload,
            )
            if response.status_code != 200:
                print(f"DEBUG: Error response body: {response.text}")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            print(f"❌ HTTP Error during generation: {e}")
            raise e

    async def generate_stream(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        """
        @definition : Génère une réponse en streaming depuis
        le service d'inférence.
        @args/params : messages (List[dict]): L'historique de la
        conversation.
        @return : AsyncGenerator[str, None] : Un générateur asynchrone
        qui produit les tokens de la réponse.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "temperature": 0.05,
            "max_tokens": 150,
            "repetition_penalty": 1.5,
        }
        print(
            f"DEBUG: Calling {self.inference_url}/v1/chat/completions "
            f"with streaming payload: {json.dumps(payload)}"
        )

        try:
            async with self.client.stream(
                "POST", f"{self.inference_url}/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        content = line[len("data:") :].strip()
                        if content != "[DONE]":
                            chunk = json.loads(content)
                            if "content" in chunk["choices"][0]["delta"]:
                                yield chunk["choices"][0]["delta"]["content"]
        except httpx.HTTPError as e:
            print(f"❌ HTTP Error during streaming generation: {e}")
            raise e

    async def close(self) -> None:
        """
        @definition : Ferme proprement le client HTTP asynchrone
        pour libérer les ressources.
        @args/params : Aucun
        @return : Aucun
        """
        await self.client.aclose()
        print("🔌 [REMOTE] Remote Inference Client closed successfully.")
