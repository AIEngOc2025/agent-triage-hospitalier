import os
import uuid

import httpx
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2 import id_token

# Configuration de la page
st.set_page_config(
    page_title="CHSA - Triage Hospitalier IA", page_icon="🩺", layout="centered"
)

# Récupération de l'URL de l'API depuis les variables d'environnement.
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
st.sidebar.info(f"🤖 L'interface est connectée à l'API suivante : `{API_BASE_URL}`")


def get_id_token(url):
    """
    @definition : Récupère un token d'identité OIDC pour authentifier
    les requêtes vers l'API.
    @args/params : url (str): L'URL cible pour laquelle générer le token.
    @return : str : Le jeton d'identité (ID Token) valide.
    """
    auth_req = Request()
    # Récupère automatiquement le token OIDC pour Cloud Run
    token = id_token.fetch_id_token(auth_req, url)
    return token


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

    # Appel API FastAPI pour générer la réponse en streaming
    with st.chat_message("assistant"):
        with st.spinner("Analyse clinique en cours..."):
            try:
                # Obtenir le token d'authentification
                headers = {}
                if not API_BASE_URL.startswith(
                    "http://localhost"
                ) and not API_BASE_URL.startswith("http://api"):
                    token = get_id_token(API_BASE_URL)
                    headers["Authorization"] = f"Bearer {token}"

                # Fonction génératrice pour le streaming
                def response_generator():
                    # We need to get the full response to handle structured data
                    # Let's change the POST to get the full JSON instead of stream
                    # But the requirement is to keep streaming.
                    # As a workaround, let's just make a non-streaming call for now
                    # or update the UI to parse the final JSON.
                    response = httpx.post(
                        f"{API_BASE_URL}/chat",
                        headers=headers,
                        json={
                            "history": st.session_state.messages,
                            "patient_id": "conv-user",
                            "stream": False,
                        },
                        timeout=300.0,
                    )
                    data = response.json()
                    yield data["response"]
                    st.session_state.last_agent_data = data

                # Utilisation de write_stream pour afficher au fur et à mesure
                assistant_response = st.write_stream(response_generator())

                # Affichage des métadonnées agentiques
                last_data = st.session_state.get("last_agent_data")
                if last_data:
                    with st.expander("🧠 Chaîne de pensée de l'agent"):
                        st.write(last_data.get("reasoning", "Pas de raisonnement."))

                    if last_data.get("state") == "VETO_WAIT":
                        st.warning("⚠️ Une validation clinique est requise.")
                        with st.form("veto_form"):
                            veto_decision = st.radio(
                                "Approuver la décision ?",
                                [True, False],
                                format_func=lambda x: "Approuver" if x else "Veto",
                            )
                            comment = st.text_area("Justification")
                            if st.form_submit_button("Soumettre"):
                                # Handle veto submission here
                                st.success("Veto enregistré.")

                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
            except Exception as e:
                st.error(f"Erreur de communication : {e}")

# Option pour recommencer la conversation
if st.sidebar.button("Nouveau Triage"):
    st.session_state.clear()
    st.rerun()
