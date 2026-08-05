import os
import sys

import requests
import streamlit as st

# Path Correction
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="CHSA - Triage Hospitalier", page_icon="🩺")

st.title("🩺 Assistant de Triage - CHSA")

# État de la session pour gérer les étapes
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.data = {}


def next_step():
    st.session_state.step += 1


if st.session_state.step == 0:
    st.session_state.data["name"] = st.text_input("Prénom :")
    st.session_state.data["age"] = st.number_input("Âge :", min_value=0, max_value=120)
    if st.button("Suivant"):
        next_step()
        st.rerun()

elif st.session_state.step == 1:
    st.session_state.data["symptoms"] = st.text_area("Décrivez vos symptômes :")
    if st.button("Analyser"):
        next_step()
        st.rerun()

elif st.session_state.step == 2:
    st.write("Analyse en cours...")
    payload = {
        "history": [
            {
                "role": "user",
                "content": f"Patient: {st.session_state.data['name']}, {st.session_state.data['age']} ans. "
                f"Symptômes: {st.session_state.data['symptoms']}",
            }
        ],
        "patient_id": "demo-triage",
        "stream": False,
    }

    try:
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        response.raise_for_status()
        st.write(response.json()["response"])
    except Exception as e:
        st.error(f"Erreur : {e}")

    if st.button("Nouveau Triage"):
        st.session_state.step = 0
        st.rerun()
