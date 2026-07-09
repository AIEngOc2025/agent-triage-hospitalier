import os
import platform
from pathlib import Path

# --- Environnement ---
# Set to True for FastAPI Cloud / GPU deployment
IS_PRODUCTION = os.getenv("IS_PRODUCTION", "False").lower() == "true"
IS_MACOS = platform.system() == "Darwin"

# --- Project Root ---
# Détermine le chemin racine du projet de manière dynamique
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Paths ---
# Les chemins sont maintenant relatifs à la racine du projet, ce qui les rend portables.
MODEL_PATH = str(PROJECT_ROOT / "models/merged_dpo_final_chsa")
LOG_FILE = str(PROJECT_ROOT / "logs/triage.log")

# --- vLLM Params ---
VLLM_MAX_MODEL_LEN = 2048
VLLM_TENSOR_PARALLEL_SIZE = 1