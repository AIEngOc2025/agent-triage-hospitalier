import uvicorn
import os
import subprocess
import sys
from pathlib import Path

# --- Configuration robuste du chemin d'accès ---
# Ajoute la racine du projet au PYTHONPATH pour garantir que les imports (ex: `from src.api...`) fonctionnent
# peu importe d'où le script est lancé.
project_root = Path(__file__).resolve().parent # This is already the project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.api.config import APP_ENTRYPOINT, API_PORT, IS_PRODUCTION  # noqa: E402

def main():
    """

    Lance le serveur d'application ASGI.
    - En développement: Utilise Uvicorn avec rechargement automatique.
    - En production: Utilise Gunicorn pour la robustesse et la performance.
    """
    if IS_PRODUCTION:
        # FastAPI Cloud fournit le port via la variable d'environnement PORT
        port = os.getenv("PORT", "8001") # Le port 8001 est standard pour les conteneurs
        workers = str(os.getenv("WEB_CONCURRENCY", "4")) # Gunicorn attend une chaîne de caractères pour le nombre de workers

        # Commande pour la production avec Gunicorn
        # -w: Lance N processus "workers"
        # -k uvicorn.workers.UvicornWorker: Utilise Uvicorn pour gérer les requêtes asynchrones
        # --preload: Charge l'application (et le modèle vLLM) avant de forker les workers
        #            pour partager la mémoire du modèle et éviter les OOM.
        command = [
            "gunicorn",
            "-w", workers,
            "-k", "uvicorn.workers.UvicornWorker",
            "--preload",
            "-b", f"0.0.0.0:{port}", # Bind to all network interfaces
            APP_ENTRYPOINT # Application entry point (e.g., "src.api.orchestrateur:app")
        ]
        print(f"🚀 Lancement du serveur de PRODUCTION avec la commande : {' '.join(command)}")
        subprocess.run(command)
    else:
        # Commande pour le développement local avec Uvicorn
        print(f"🚀 Lancement du serveur de DÉVELOPPEMENT sur http://127.0.0.1:{API_PORT}")
        # Uvicorn gère le rechargement automatique, idéal pour le développement.
        uvicorn.run(APP_ENTRYPOINT, host="127.0.0.1", port=API_PORT, reload=True)

if __name__ == "__main__":
    main()