import logging
from typing import AsyncGenerator, List
from app.schemas import TriageResponse

logger = logging.getLogger(__name__)


class LocalEngine:
    def __init__(self, settings):
        self.settings = settings
        self.engine_type = "LocalMLX"
        self.model = None
        self.tokenizer = None

    def initialize(self):
        try:
            import mlx_lm
        except ImportError:
            logger.warning("⚠️ mlx_lm not found. LocalEngine initialized in dummy mode.")
            return

        print("🏠 [LOCAL] Initializing local MLX engine...")
        try:
            self.model, self.tokenizer = mlx_lm.load(self.settings.MODEL_PATH)
        except Exception as e:
            logger.error(f"❌ MLX loading failed: {e}")
            raise

    async def generate_stream(
        self, messages: List[dict], request_id: str
    ) -> AsyncGenerator[str, None]:
        if self.model is None:
            raise RuntimeError("MLX engine not initialized (mlx_lm missing).")
        
        import mlx_lm

        # Convert messages to prompt string for mlx_lm
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = mlx_lm.generate(
            self.model, self.tokenizer, prompt=prompt, verbose=False
        )
        yield response

    async def generate(self, messages: List[dict]) -> str:
        if self.model is None:
            raise RuntimeError("MLX engine not initialized (mlx_lm missing).")
        
        import mlx_lm

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return mlx_lm.generate(self.model, self.tokenizer, prompt=prompt, verbose=False)

    async def generate_structured(self, messages: List[dict]) -> TriageResponse:
        if self.model is None:
            raise RuntimeError("MLX engine not initialized (mlx_lm missing).")
            
        import mlx_lm

        """
        Génère du texte brut, nettoie le JSON, puis valide avec Pydantic.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 1. Génération du texte brut
        raw_text = mlx_lm.generate(
            self.model, self.tokenizer, prompt=prompt, verbose=False
        )

        # 2. Nettoyage basique (au cas où le modèle ajoute des ```json ...)
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()

        # 3. Validation par Pydantic
        try:
            return TriageResponse.model_validate_json(clean_text)
        except Exception as e:
            logger.error(f"❌ Échec parsing JSON local : {e}\nRaw: {raw_text}")
            raise ValueError(f"Le modèle local n'a pas retourné de JSON valide: {e}")

    async def close(self):
        pass
