from app.settings import settings
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_inference():
    """
    @definition : Test qualitatif du modèle fusionné DPO sur un cas de triage.
    @return : None
    """
    print(f"📥 Chargement du modèle depuis : {settings.MODEL_PATH}")

    # Utilisation de transformers pour une compatibilité maximale (CPU/GPU/Mac)
    try:
        model = AutoModelForCausalLM.from_pretrained(str(settings.MODEL_PATH), trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(str(settings.MODEL_PATH), trust_remote_code=True)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle avec transformers : {e}")
        raise

    # Cas de test : Infarctus suspecté (Urgence Vitale)
    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant virtuel expert en triage hospitalier. Ton rôle est d'analyser les symptômes et de recommander une priorité (Urgences, Médecine Générale, ou Soins à domicile).",
        },
        {
            "role": "user",
            "content": "Un patient de 45 ans arrive avec une douleur thoracique intense irradiant vers le bras gauche et des sueurs froides depuis 20 minutes.",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print("\n🚀 Génération de la réponse...")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=512)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # La réponse contient souvent le prompt, nous l'enlevons pour plus de clarté.
    response = response_text.replace(prompt, "").strip()

    print("\n--- RÉPONSE DU MODÈLE ---")
    print(response)
    print("-------------------------\n")


if __name__ == "__main__":
    test_inference()
