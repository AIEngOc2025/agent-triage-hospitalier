import os
from pathlib import Path

# --- Détection de l'environnement ---
IS_PRODUCTION = os.getenv("FASTAPI_CLOUD", "false").lower() == "true"

# --- 1. Project Structure ---
# Le chemin de base est la racine du projet (un niveau au-dessus de 'src')
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- 2. vLLM Engine Configuration ---
if IS_PRODUCTION:
    # Dans l'environnement FastAPI Cloud, les binaires sont dans le PATH du venv
    VENV_PYTHON_PATH = "python"
    VLLM_BINARY_PATH = "vllm"
    # Le modèle est copié dans le conteneur
    MODEL_PATH = "models/merged_dpo_final_chsa"
else:
    # Chemins pour le développement local sur Mac
    VENV_PYTHON_PATH = os.getenv("VENV_PYTHON_PATH", "/Users/mpaga/.venv-vllm-metal/bin/python")
    VLLM_BINARY_PATH = os.getenv("VLLM_BINARY_PATH", "/Users/mpaga/.venv-vllm-metal/bin/vllm")
    # Chemin absolu vers le modèle pour éviter les erreurs d'interprétation par vLLM
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "merged_dpo_final_chsa"))


# Network ports
VLLM_PORT = 8000
API_PORT = 8001
VLLM_HOST = "127.0.0.1"

# vLLM server arguments
VLLM_SERVER_ARGS = [
    "--host", VLLM_HOST,
    "--port", str(VLLM_PORT),
    #"--device", "mps",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.7",
]

# --- 3. Logging & Auditing ---
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "audit_medical.jsonl"
VLLM_LOG_FILE = LOG_DIR / "vllm_server.log"