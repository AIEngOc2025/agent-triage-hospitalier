import os
from pathlib import Path
import sys
import tempfile

# --- Environment Detection ---
# Detect production environment for FastAPI Cloud or Hugging Face Spaces
IS_PRODUCTION = (
    os.getenv("APP_ENV", "development").lower() == "production"
    or os.getenv("FASTAPI_CLOUD", "false").lower() == "true"
    or os.getenv("SPACE_ID") is not None
)

# Détection de l'environnement pour l'accélération matérielle
IS_MACOS = sys.platform == "darwin"

# --- 1. Project Structure ---
# BASE_DIR est la racine du projet (le parent du dossier 'app')
# __file__ -> /app/api/config.py -> .parent -> /app/api -> .parent -> /app -> .parent -> /
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# --- 2. vLLM Engine Configuration ---
# Le chemin du modèle est maintenant construit de manière cohérente pour tous les environnements.
# En production (conteneur), le chemin sera absolu, par ex. /app/models/merged_dpo_final_chsa
MODEL_PATH = str(MODELS_DIR / "merged_dpo_final_chsa")

VLLM_MAX_MODEL_LEN = 4096

# Pour le déploiement multi-GPU (ex: sur Hugging Face Spaces avec 2xT4)
VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))

# NOTE: Les variables ci-dessous sont conservées pour information mais ne sont pas
# utilisées par l'orchestrateur qui instancie LLM directement. Elles seraient
# utiles si vous lanciez vLLM comme un serveur API séparé.
# VLLM_HOST = "127.0.0.1"
# VLLM_PORT = 8003
# API_PORT = 8004
# APP_ENTRYPOINT = "app.api.main:app"

# --- 3. Logging & Auditing ---
# Utilise un dossier temporaire système pour les logs, ce qui est robuste.
LOG_DIR = Path(tempfile.gettempdir())
LOG_FILE = LOG_DIR / "audit_medical.jsonl"
