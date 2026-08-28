import asyncio
import logging
from typing import Any, Callable, TypeVar

from httpx import ConnectError, HTTPStatusError, TimeoutException

logger = logging.getLogger(__name__)

# Paramètres de retry (déplacer dans settings si nécessaire)
MAX_RETRIES = 3
BASE_BACKOFF_SEC = 1.0
RETRYABLE_STATUS_CODES = {502, 503, 504}
RETRYABLE_EXCEPTIONS = (HTTPStatusError, ConnectError, TimeoutException)

T = TypeVar("T")


async def call_with_retry(func: Callable[[], Any]) -> Any:
    """
    @definition : Exécute une fonction asynchrone avec retry exponentiel
        sur les erreurs 5xx retryables et timeouts.
    @args/params : func (Callable[[], Any]) - Fonction asynchrone à appeler.
    @return : Any - Résultat de la fonction exécutée.
    """
    retries = 0
    while True:
        try:
            return await func()
        except RETRYABLE_EXCEPTIONS as e:
            # Vérifier si l'erreur HTTP est retryable
            if isinstance(e, HTTPStatusError):
                status_code = getattr(e, "response", None) and e.response.status_code
                if status_code not in RETRYABLE_STATUS_CODES:
                    raise e

            if retries >= MAX_RETRIES:
                logger.error("❌ Max retries exhausted. Failing.")
                raise e

            retries += 1
            wait = BASE_BACKOFF_SEC * (2 ** (retries - 1))
            logger.warning(f"⚠️ Retry {retries}/{MAX_RETRIES} after {wait}s due to: {e}")
            await asyncio.sleep(wait)
        except Exception as e:
            # Erreurs non-retryables
            raise e
