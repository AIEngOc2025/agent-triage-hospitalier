import asyncio
import logging
from typing import Any, Callable, TypeVar

from httpx import ConnectError, HTTPStatusError

logger = logging.getLogger(__name__)

# Paramètres de retry (déplacer dans settings si nécessaire)
MAX_RETRIES = 3
BASE_BACKOFF_SEC = 1.0
RETRYABLE_STATUS_CODES = {502, 503, 504}

T = TypeVar("T")


async def call_with_retry(func: Callable[[], Any]) -> Any:
    """
    @definition : Exécute une fonction asynchrone avec retry exponentiel
        sur les erreurs 5xx retryables.
    @args/params : func (Callable[[], Any]) - Fonction asynchrone à appeler.
    @return : Any - Résultat de la fonction exécutée.
    """
    retries = 0
    while True:
        try:
            return await func()
        except (HTTPStatusError, ConnectError) as e:
            # Vérifier si l'erreur est retryable
            status_code = getattr(e, "response", None) and e.response.status_code
            if (
                isinstance(e, HTTPStatusError)
                and status_code not in RETRYABLE_STATUS_CODES
            ):
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
