import os

import pytest
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B"
ADAPTERS = "models/sft_final_chsa"

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_MODEL_TESTS"),
    reason="Tests de chargement de modèles désactivés en CI; définissez RUN_MODEL_TESTS=1 pour les activer.",
)


def test_cpu_smoke():
    if not os.path.isdir(ADAPTERS):
        pytest.skip(f"Dossier d'adaptateurs introuvable : {ADAPTERS}")

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map={"": device},
    )
    model = PeftModel.from_pretrained(base_model, ADAPTERS)

    messages = [
        {"role": "system", "content": "Tu es l'infirmier d'accueil du CHSA."},
        {"role": "user", "content": "Bonjour, j'ai une douleur à la poitrine."},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        repetition_penalty=1.5,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert isinstance(decoded, str)
    assert decoded
