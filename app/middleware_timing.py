import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """
        @definition: Middleware pour mesurer la latence réseau globale d'une requête.
        @args/params:
            - request (Request): Requête FastAPI.
            - call_next: Callback vers le prochain middleware/endpoint.
        @return: Réponse HTTP.
        """
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Ajout de la latence totale dans les headers pour visibilité
        response.headers["X-Process-Time-Ms"] = str(round(duration * 1000, 2))
        return response
