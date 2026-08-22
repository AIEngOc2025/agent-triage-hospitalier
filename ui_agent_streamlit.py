import streamlit as st
import httpx
import uuid

# Configuration
API_URL = "https://agent-api-gateway-414294705487.europe-west1.run.app/chat"
st.set_page_config(page_title="CHSA - Agent Triage", page_icon="🩺")
st.title("🩺 Agent de Triage Hospitalier (Client Distant)")

# Initialisation de la session
if "patient_id" not in st.session_state:
    st.session_state.patient_id = str(uuid.uuid4())
    st.session_state.messages = []

# Affichage de l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Entrée utilisateur
if prompt := st.chat_input("Décrivez la plainte du patient :"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Appel à l'API distante
    with st.chat_message("assistant"):
        with st.spinner("Analyse clinique..."):
            try:
                payload = {
                    "history": st.session_state.messages,
                    "patient_id": st.session_state.patient_id,
                    "stream": False,
                }
                response = httpx.post(API_URL, json=payload, timeout=60.0)
                response.raise_for_status()
                result = response.json()

                # Récupération de la réponse
                msg = result.get("response", "Pas de réponse reçue.")
                st.write(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            except Exception as e:
                st.error(f"Erreur de connexion à l'API : {e}")
