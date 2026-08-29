import os
import random
import uuid

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "300"))
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "2000"))

st.set_page_config(
    page_title="CHSA - Agent de Triage Médical",
    page_icon="🩺",
    layout="wide",
)


def get_triage_badge(level: str) -> str:
    """
    @definition : Génère un badge Markdown coloré pour le niveau de priorité.
    @args/params : level (str) - Niveau de triage ('maximale', 'modérée', 'différée').
    @return : str - Chaîne formatée en Markdown avec emoji et code couleur.
    """
    clean_lvl = (level or "").lower()
    if "max" in clean_lvl:
        return "🔴 **PRIORITÉ MAXIMALE (Urgence vitale immédiate)**"
    if "mod" in clean_lvl:
        return "🟡 **PRIORITÉ MODÉRÉE (Prise en charge sous 1h à 2h)**"
    if "diff" in clean_lvl:
        return "🟢 **PRIORITÉ DIFFÉRÉE (Soins programmés / Médecine générale)**"
    return "⚪ **Évaluation en cours**"


# Initialisation de la session
if "patient_id" not in st.session_state:
    st.session_state.patient_id = f"PAT-{random.randint(100, 999999)}"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour. Je suis l'assistant IA de triage du Centre Hospitalier "
                "Saint-Aurélien (CHSA).\n\n"
                "Quel est le motif de votre venue ou quels symptômes ressentez-vous ?"
            ),
        }
    ]
if "last_metadata" not in st.session_state:
    st.session_state.last_metadata = {}

# --- Barre latérale : Paramètres & Métadonnées ---
with st.sidebar:
    st.header("🏥 CHSA - Triage Médical")
    st.caption("Système d'aide à la décision pour le personnel soignant.")
    st.divider()

    st.markdown(f"**Identifiant Patient :** `{st.session_state.patient_id}`")
    st.markdown(f"**Passerelle API :** `{API_BASE_URL}`")

    stream_mode = st.toggle("Activer le streaming", value=False)

    if st.session_state.last_metadata.get("triage_level"):
        st.divider()
        st.subheader("Orientation Clinique")
        lvl = st.session_state.last_metadata.get("triage_level")
        st.markdown(get_triage_badge(lvl))

        if st.session_state.last_metadata.get("is_emergency"):
            st.error("🚨 DRAPEAU ROUGE DÉTECTÉ - Alerte soignant requise.")

    st.divider()
    if st.button("🔄 Nouveau Triage (Réinitialiser)", use_container_width=True):
        st.session_state.clear()
        st.session_state.patient_id = f"PAT-{random.randint(100, 999999)}"
        st.rerun()

# --- Zone Principale ---
st.title("🩺 Assistant Conversationnel de Triage - CHSA")
st.caption(
    "Recueil interactif des symptômes avec anonymisation RGPD et garde-fous cliniques."
)


# Affichage de l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Saisie utilisateur
if user_input := st.chat_input("Décrivez vos symptômes, douleur ou malaise..."):
    if len(user_input) > MAX_INPUT_LENGTH:
        st.warning(
            f"Votre message dépasse la limite autorisée "
            f"({MAX_INPUT_LENGTH} caractères)."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        payload = {
            "history": st.session_state.messages,
            "patient_id": st.session_state.patient_id,
            "stream": stream_mode,
        }

        if stream_mode:
            try:

                def response_stream():
                    with httpx.stream(
                        "POST",
                        f"{API_BASE_URL}/chat",
                        json=payload,
                        timeout=API_TIMEOUT,
                    ) as response:
                        response.raise_for_status()
                        for line in response.iter_text():
                            yield line

                assistant_response = st.write_stream(response_stream())
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
            except Exception as e:
                st.error(f"Erreur de communication en streaming : {e}")
        else:
            with st.spinner("Analyse clinique et classification en cours..."):
                try:
                    with httpx.Client(timeout=API_TIMEOUT) as client:
                        response = client.post(f"{API_BASE_URL}/chat", json=payload)
                    response.raise_for_status()
                    data = response.json()

                    assistant_response = data.get("response", "Pas de réponse.")
                    st.write(assistant_response)

                    # Sauvegarde des métadonnées
                    st.session_state.last_metadata = {
                        "triage_level": data.get("triage_level"),
                        "reasoning": data.get("reasoning"),
                        "state": data.get("state"),
                        "audit_ref": data.get("audit_ref"),
                        "is_emergency": data.get("is_emergency"),
                    }

                    # Badge sous le message
                    if data.get("triage_level"):
                        st.markdown(get_triage_badge(data.get("triage_level")))

                    with st.expander("🔍 Métadonnées & Auditabilité"):
                        st.write(f"**État agentique :** `{data.get('state')}`")
                        st.write(f"**Raisonnement :** {data.get('reasoning')}")
                        st.write(f"**Réf Audit RGPD :** `{data.get('audit_ref')}`")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_response}
                    )
                except httpx.HTTPError as e:
                    resp = getattr(e, "response", None)
                    code = resp.status_code if resp is not None else "timeout"
                    st.error(f"Erreur de communication avec le serveur ({code}).")
                except Exception as e:
                    st.error(f"Une erreur inattendue est survenue : {e}")
