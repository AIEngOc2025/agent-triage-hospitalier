import time
from functools import wraps
from typing import Any, Callable


def time_execution(component_name: str):
    """
    @definition: Décorateur pour mesurer le temps d'exécution d'une fonction asynchrone
    et stocker la latence dans un attribut de la fonction.
    @args/params:
        - component_name (str): Nom du composant mesuré.
    @return: Fonction décorée.
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            duration = time.perf_counter() - start

            # Stockage de la latence sur la fonction décorée
            if not hasattr(wrapper, "last_latency"):
                wrapper.last_latency = 0.0
            wrapper.last_latency = duration
            return result

        return wrapper

    return decorator
