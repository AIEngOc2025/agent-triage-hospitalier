import streamlit as st

from agent.orchestrator import TriageAgentOrchestrator

# Configuration
st.set_page_config(page_title="CHSA - Agent Triage", page_icon="🩺")
st.title("🩺 Agent de Triage Hospitalier (Dialogue Itératif)")

# Initialisation de l'agent dans la session
if "agent" not in st.session_state:
    st.session_state.agent = TriageAgentOrchestrator()
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

    # Exécution de l'agent
    with st.chat_message("assistant"):
        with st.spinner("Analyse clinique..."):
            result = st.session_state.agent.run(prompt)

            if result["status"] == "PENDING_CLARIFICATION":
                msg = result["question"]
                st.write(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            elif result["status"] == "AUTO_FINALIZED":
                msg = f"Triage finalisé : **{result['final_decision']}**. {result['comment']}"
                st.write(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            elif result["status"] == "PENDING_VETO":
                msg = f"Recommandation : **{result['recommended_level']}**. Validation requise."
                st.write(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                # Gérer le veto ici (simplifié pour le POC)
                st.warning("⚠️ Veto clinique à implémenter dans cette UI.")
