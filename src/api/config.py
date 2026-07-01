import os
from pathlib import Path

# --- Environment Detection ---
# Detect production environment for FastAPI Cloud or Hugging Face Spaces
IS_PRODUCTION = (os.getenv("APP_ENV", "development").lower() == "production" or
                 os.getenv("FASTAPI_CLOUD", "false").lower() == "true" or
                 os.getenv("SPACE_ID") is not None)


# --- 1. Project Structure ---
SRC_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = SRC_DIR.parent
MODELS_DIR = BASE_DIR / "models" # Le dossier models est à la racine du projet

# --- 2. vLLM Engine Configuration ---
if IS_PRODUCTION:
    # In the FastAPI Cloud environment, binaries are in the venv's PATH
    VLLM_BINARY = "vllm"
    # The model is copied into the container at /app/models/
    MODEL_PATH = str(BASE_DIR / "models" / "merged_dpo_final_chsa")  # Chemin absolu dans le conteneur
else:
    # Paths for local development on Mac
    VLLM_BINARY = os.getenv("VLLM_BINARY_PATH", "/Users/mpaga/.venv-vllm-metal/bin/vllm")
    # Absolute path to the model to avoid interpretation errors by vLLM
    MODEL_PATH = str(MODELS_DIR / "merged_dpo_final_chsa") # Chemin absolu local

# L'entrypoint est maintenant le même pour le local et la production
APP_ENTRYPOINT = "src.api.orchestrateur:app"

# Network ports
VLLM_PORT = 8003
API_PORT = 8004
VLLM_HOST = "127.0.0.1"

# vLLM server arguments
# Note: Ces arguments sont pour le mode 'local' (main.py), pas pour l'orchestrateur.
VLLM_SERVER_ARGS = [
    "--model", MODEL_PATH,
    "--host", VLLM_HOST,
    "--port", str(VLLM_PORT),
   # "--max-model-len", "4096",
    #"--gpu-memory-utilization", "0.7",
]

# --- 3. Logging & Auditing ---
import tempfile
LOG_DIR = Path(tempfile.gettempdir())
LOG_FILE = LOG_DIR / "audit_medical.jsonl"
VLLM_LOG_FILE = LOG_DIR / "vllm_server.log"