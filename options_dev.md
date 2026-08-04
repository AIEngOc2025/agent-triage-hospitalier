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

## 4. Inference & Model Behavior Optimization
- **Observed Issues:** Current local inference (using `transformers`) shows the model formulating medical diagnostics (violating system prompt) and entering repetition loops.
- **Needed Actions:**
  - Configure generation parameters (`repetition_penalty`, `max_new_tokens`, `stopping_criteria`) in the inference client.
  - Further align the model via SFT/DPO to strictly adhere to triage-only, non-diagnostic constraints.
  - Validate performance again using the same test script after adjustments.

## 5. Network Configuration & Validation
- **Current Status:** `agent-inference-service` is set to `ingress internal` to comply with security requirements.
- **Validation Issues:** Public `curl` requests to the inference service fail with 404 (expected security behavior).
- **Needed Actions:**
  - Validate inter-service communication (API Gateway -> Inference Service) using a testing tool within the VPC (e.g., Cloud Function or VM within the same network).
  - Verify IAM permissions (`Cloud Run Invoker`) for the API Gateway service account.
