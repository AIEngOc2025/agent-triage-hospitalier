import os
import pytest
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
SFT_ADAPTERS = "models/sft"
DEVICE = "cpu"

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_MODEL_TESTS"),
    reason="Tests de chargement de modèles désactivés en CI; définissez RUN_MODEL_TESTS=1 pour les activer.",
)


def generate_response(query):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, SFT_ADAPTERS)
    model.eval()

    messages = [
        {"role": "system", "content": "Tu es l'infirmier d'accueil bienveillant du CHSA."},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in full_text:
        return full_text.split("assistant")[-1].strip()
    return full_text


def test_sft_model_smoke():
    if not os.path.isdir(SFT_ADAPTERS):
        pytest.skip(f"Dossier d'adaptateurs introuvable : {SFT_ADAPTERS}")

    query = "Bonjour, j'ai une douleur très vive dans la poitrine et mon bras gauche est engourdi. Que faut-il faire ?"
    response = generate_response(query)
    assert isinstance(response, str)
    assert response
