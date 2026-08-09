from typing import List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    patient_id: str = Field(
        ..., pattern=r"^PAT-\d{3,}$", description="Patient identifier (format: PAT-XXX)"
    )
    history: List[dict] = Field(..., min_length=1, max_length=50)
    stream: bool = False


# Example of a valid request
valid_data = {
    "patient_id": "PAT-123",
    "history": [{"role": "user", "content": "Bonjour"}],
    "stream": False,
}

# Example of an invalid request (e.g., wrong pattern for patient_id)
invalid_data = {
    "patient_id": "123",
    "history": [{"role": "user", "content": "Bonjour"}],
    "stream": False,
}

try:
    ChatRequest(**valid_data)
    print("Valid data passed")
except Exception as e:
    print(f"Valid data failed: {e}")

try:
    ChatRequest(**invalid_data)
    print("Invalid data passed (unexpected)")
except Exception as e:
    print(f"Invalid data correctly failed: {e}")
