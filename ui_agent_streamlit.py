import os
import random

import httpx
import streamlit as st

# Configuration
# En production Cloud Run, surcharger via la variable d'env API_URL.
API_URL = os.getenv(
    "API_URL",
    "https://agent-api-gateway-414294705487.europe-west1.run.app/chat",
)
st.set_page_config(page_title="CHSA - Agent Triage", page_icon="🩺")
st.title("🩺 Agent de Triage Hospitalier (Client Distant)")
st.caption(f"Patient ID de la session : `{st.session_state.get('patient_id', '—')}`")
st.caption(f"API cible : `{API_URL}`")

# Initialisation de la session
# L'API attend un patient_id conforme au pattern Pydantic
# ^(PAT-\d{3,}|conv-user)$ (cf. app/main.py / app/schemas.py).
if "patient_id" not in st.session_state:
    st.session_state.patient_id = f"PAT-{random.randint(100, 999_999_999)}"
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
                response = httpx.post(API_URL, json=payload, timeout=360.0)
                response.raise_for_status()
                result = response.json()

                # Récupération de la réponse et du raisonnement
                msg = result.get("response", "Pas de réponse reçue.")
                reasoning = result.get("reasoning", None)

                if reasoning:
                    with st.expander("🧠 Chaîne de pensée de l'agent"):
                        st.write(reasoning)

                st.write(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            except httpx.HTTPStatusError as e:
                st.error(
                    f"Erreur API ({e.response.status_code}) : {e.response.text[:300]}"
                )
            except Exception as e:
                st.error(f"Erreur de connexion à l'API : {e}")
