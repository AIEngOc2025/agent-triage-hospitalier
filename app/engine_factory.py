from app.core.settings import settings
from app.local.engine import LocalEngine
from app.remote.engine import RemoteEngine

def get_engine():
    if settings.ENGINE_MODE == "local":
        return LocalEngine(settings)
    return RemoteEngine()

engine = get_engine()
