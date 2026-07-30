import uuid

import httpx
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="CHSA - Triage Hospitalier IA", page_icon="🩺", layout="centered"
)

# Configuration de l'URL de l'API dans la barre latérale
API_BASE_URL = st.sidebar.text_input(
    "URL de l'API Gateway",
    value="https://agent-triage-hospitalier-rlgcjqsysq-ew.a.run.app",
)

# Initialisation de l'état de la session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    # Premier message d'accueil de l'assistant
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour ! Je suis l'assistant virtuel de triage du CHSA. "
                "Pour mieux vous orienter, pouvez-vous me dire votre prénom, "
                "votre âge, et ce qui vous amène aujourd'hui ?"
            ),
        }
    ]

# Titre principal
st.title("🩺 Assistant de Triage Hospitalier - CHSA")
st.write(
    "Discutez directement avec notre assistant intelligent pour évaluer "
    "votre situation."
)

# Étape unique : Le Chat conversationnel direct
st.subheader("💬 Discussion de Triage")

# Affichage de l'historique des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Zone de saisie utilisateur
if user_input := st.chat_input("Répondez à l'assistant ou décrivez votre situation..."):
    # Afficher le message de l'utilisateur
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Appel API FastAPI pour générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Analyse clinique en cours..."):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "history": st.session_state.messages,
                        "patient_id": "conv-user",  # ID anonyme pour chat direct
                        "stream": False
                    },
                    timeout=300.0
                )


                if response.status_code == 200:
                    assistant_response = response.json()["response"]
                    st.write(assistant_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_response}
                    )
                else:
                    st.error("Erreur de connexion au serveur de triage.")
            except Exception as e:
                st.error(f"Erreur de communication : {e}")

# Option pour recommencer la conversation
if st.sidebar.button("Nouveau Triage"):
    st.session_state.clear()
    st.rerun()
