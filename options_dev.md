# Options for Development - Tomorrow's Session

## 1. Deployment of Streamlit Frontend
- Currently, only the API is deployed on Cloud Run.
- Streamlit needs to be hosted (e.g., Streamlit Community Cloud or dedicated Cloud Run service) to expose the triage interface to users.

## 2. Enhancement of Triage Logic
- Refine system prompts in `app-service/app/system_prompts.py` for better triage accuracy.
- Improve training scripts in `scripts/training/` if model retraining is required.

## 3. Audit & Logs Privacy (RGPD)
- Enhance anonymization logic in `app-service/app/main.py`.
- Ensure logs conform to RGPD standards.
