import os
import platform

# --- Environnement ---
# Set to True for FastAPI Cloud / GPU deployment
IS_PRODUCTION = os.getenv("IS_PRODUCTION", "False").lower() == "true"
IS_MACOS = platform.system() == "Darwin"

# --- Paths ---
# Chemin absolu vers le modèle fusionné
MODEL_PATH = "/Users/mpaga/OC/agent-triage-hospitalier/models/merged_dpo_final_chsa"
LOG_FILE = "logs/triage.log"

# --- vLLM Params ---
VLLM_MAX_MODEL_LEN = 2048
VLLM_TENSOR_PARALLEL_SIZE = 1