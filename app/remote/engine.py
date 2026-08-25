from typing import AsyncGenerator, List
from app.remote.client import RemoteInferenceClient
from app.schemas import TriageResponse
from app.api_utils import clean_response


class RemoteEngine:
    def __init__(self):
        self.client = None
        self.engine_type = "RemoteInference"

    def initialize(self):
        print("🌐 [REMOTE] Initializing remote inference client...")
        self.client = RemoteInferenceClient()

    async def generate_stream(
        self, messages: List[dict], request_id: str
    ) -> AsyncGenerator[str, None]:
        async for chunk in self.client.generate_stream(messages):
            yield chunk

    async def generate(self, messages: List[dict]) -> str:
        response = await self.client.generate(messages)
        return clean_response(response)

    async def generate_structured(self, messages: List[dict]) -> TriageResponse:
        return await self.client.generate_structured(messages)

    async def close(self):
        if self.client:
            await self.client.close()
