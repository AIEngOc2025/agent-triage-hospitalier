import os
import uuid

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "2000"))

st.set_page_config(page_title="CHSA - Triage Hospitalier", page_icon="🩺")

st.title("🩺 Assistant de Triage - CHSA")

# État de la session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour, je suis l'assistant de triage du CHSA. "
                "Comment vous sentez-vous ?"
            ),
        }
    ]

# Affichage des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Zone de saisie
if user_input := st.chat_input("Décrivez vos symptômes..."):
    if len(user_input) > MAX_INPUT_LENGTH:
        st.warning(
            f"Votre message dépasse la limite "
            f"({MAX_INPUT_LENGTH} caractères). Veuillez le raccourcir."
        )
        st.stop()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            try:
                payload = {
                    "history": st.session_state.messages,
                    "patient_id": st.session_state.session_id,
                    "stream": False,
                }
                with httpx.Client(timeout=API_TIMEOUT) as client:
                    response = client.post(f"{API_BASE_URL}/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                assistant_response = data.get("response")
                if assistant_response is None:
                    st.error("Réponse invalide du serveur.")
                    st.stop()

                st.write(assistant_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
            except httpx.HTTPError as e:
                status = e.response.status_code if e.response else "timeout"
                st.error(f"Erreur de communication avec le serveur ({status}).")
            except Exception:
                st.error("Une erreur inattendue est survenue.")

if st.sidebar.button("Nouveau Triage"):
    st.session_state.clear()
    st.rerun()
