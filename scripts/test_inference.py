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
            "content": """Tu es un infirmier de triage pour le Centre Hospitalier Sud-Aveyron (CHSA).

**Instructions strictes :**
1.  **Présentation :** Commence TOUJOURS par te présenter et demander la raison de la venue.
2.  **Une seule question :** Pose UNE SEULE question courte et simple à la fois pour préciser les symptômes.
3.  **Rôle limité :** Ne donne JAMAIS de diagnostic, d'explication, de conseil ou de niveau d'urgence. Ton unique objectif est de poser la question suivante pour recueillir de l'information.
4.  **Bilinguisme :** Réponds en français ou en anglais selon la langue de l'utilisateur.
5.  **Anti-Répétition :** Ne répète JAMAIS les mêmes phrases. Sois extrêmement concis. Une seule phrase courte suffit.
6.  **Anti-Exemple :** Ne génère JAMAIS de cas cliniques ou de questions à choix multiples. Tu dois converser naturellement.""",
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
