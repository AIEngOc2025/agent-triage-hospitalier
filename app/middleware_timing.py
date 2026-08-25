import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        response = await call_next(request)

        process_time = (time.perf_counter() - start_time) * 1000

        # Ajout du header personnalisé pour la visibilité côté client
        response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"

        logger.info(
            f"⏱️  [METRIC] Global Request: {request.method} {request.url.path} "
            f"- Latency: {process_time:.2f}ms"
        )

        return response
