import functools
import logging
import time
from contextlib import contextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@contextmanager
def measure_latency(label: str):
    """
    Context manager pour mesurer la latence d'un bloc de code.
    Log le temps écoulé en millisecondes.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"⏱️  [METRIC] {label}: {elapsed_ms:.2f}ms")


def time_execution(label: Optional[str] = None):
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal label
            if label is None:
                label = func.__name__

            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info(f"⏱️  [METRIC] {label}: {elapsed_ms:.2f}ms")

        return wrapper

    return decorator
