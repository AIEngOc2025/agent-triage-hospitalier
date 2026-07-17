from transformers import AutoModelForCausalLM, AutoTokenizer

from app.settings import settings


def test_inference():
    """
    @definition : Test qualitatif du modèle fusionné DPO sur un cas de triage.
    @return : None
    """
    print(f"📥 Chargement du modèle depuis : {settings.MODEL_PATH}")

    # Utilisation de transformers pour une compatibilité maximale (CPU/GPU/Mac)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(settings.MODEL_PATH), trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(settings.MODEL_PATH), trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle avec transformers : {e}")
        raise

    # Cas de test : Infarctus suspecté (Urgence Vitale)
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un infirmier de triage pour le Centre Hospitalier Sud-Aveyron (CHSA).\n\n"
                "**Instructions strictes :**\n"
                "1. **Présentation :** Présente-toi et demande la raison de la venue.\n"
                "2. **Une seule question :** Pose une seule question courte à la fois.\n"
                "3. **Rôle limité :** Ne donne aucun diagnostic ni conseil.\n"
                "4. **Bilinguisme :** Réponds en français ou en anglais.\n"
                "5. **Anti-Répétition :** Sois concis. Une phrase suffit.\n"
                "6. **Anti-Exemple :** Pas de cas cliniques ni de QCM."
            ),
        },
        {
            "role": "user",
            "content": "Un patient de 45 ans arrive avec une douleur thoracique "
            "intense irradiant vers le bras gauche et des sueurs "
            "froides depuis 20 minutes.",
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
