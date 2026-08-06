import json
import logging
import os
from typing import AsyncGenerator, List, Optional

import httpx

# Configure logging for this module
logger = logging.getLogger(__name__)


class RemoteInferenceClient:
    """
    Client for interacting with a remote inference service.
    """

    def __init__(
        self,
        inference_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        timeout: float = 300.0,
    ):
        """
        Initializes the remote inference client with its configurations
        and the HTTP client.
        """

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
        self.client = httpx.AsyncClient(timeout=timeout)
        logger.info(
            f"✅ [REMOTE] Remote Inference Client initialized at "
            f"{self.inference_url} with model {self.model_name}. "
            f"Params: temp={self.temperature}, "
            f"max_tokens={self.max_tokens}, "
            f"penalty={self.repetition_penalty}"
        )

    def _prepare_payload(self, messages: List[dict], stream: bool) -> dict:
        """Prepares the common payload for inference requests."""
        # Regex pour forcer : [Niveau: <maximale|modérée|différée>] - Orientation : <orientation>
        # Note: on utilise .* pour l'orientation
        regex_pattern = r"\[Niveau: (maximale|modérée|différée)\] - Orientation : .*"

        return {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "extra_body": {"guided_regex": regex_pattern},
        }

    async def generate(self, messages: List[dict]) -> str:
        """
        Generates a complete (non-streaming) response from the
        inference service.

        @args/params:
            - messages (List[dict]): The conversation history.
        @return: The textual content of the generated response (str).
        """
        payload = self._prepare_payload(messages, stream=False)
        logger.debug(
            f"Calling {self.inference_url}/v1/chat/completions with "
            f"payload: {json.dumps(payload)}"
        )

        try:
            response = await self.client.post(
                f"{self.inference_url}/v1/chat/completions",
                json=payload,
            )
            if response.status_code != 200:
                logger.error(f"Error response body: {response.text}")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            logger.error(f"❌ HTTP Error during generation: {e}")
            raise e

    async def generate_stream(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        """
        Generates a streaming response from the inference service.

        @args/params:
            - messages (List[dict]): The conversation history.
        @return: An asynchronous generator that yields the response tokens
            (AsyncGenerator[str, None]).
        """
        payload = self._prepare_payload(messages, stream=True)
        logger.debug(
            f"Calling {self.inference_url}/v1/chat/completions with "
            f"streaming payload: {json.dumps(payload)}"
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
            raise e

    async def close(self) -> None:
        """
        Closes the asynchronous HTTP client cleanly to release resources.

        @args/params: None
        @return: None
        """
        await self.client.aclose()
        logger.info("🔌 [REMOTE] Remote Inference Client closed successfully.")
