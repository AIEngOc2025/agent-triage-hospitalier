import time


def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def show_slide(title, content):
    print(f"\n>> {title}")
    for line in content:
        print(f"   - {line}")
    time.sleep(1)


def main():
    print_header("CHSA AI Triage Agent - Soutenance Technique")

    show_slide(
        "Alignement avec les Livrables",
        [
            "Dataset médical : Structuration et versionnage (format JSONL/Hugging Face).",
            "Modèle IA : Fine-tuning SFT + Alignement DPO (vLLM).",
            "Déploiement : Architecture Cloud Run optimisée (inférence rapide).",
            "Pipeline CI/CD : Automatisation complète via Cloud Build.",
        ],
    )

    show_slide(
        "Architecture de l'API (app/main.py)",
        [
            "Polymorphisme Engine : Gestion fine entre chat libre et structuré.",
            "Endpoints Découplés : /chat (streaming) vs /triage (JSON typé).",
            "Guardrails : Détection déterministe des urgences vitales.",
            "Conformité : Audit anonymisé et monitoring de latence intégré.",
        ],
    )

    show_slide(
        "Stratégie de Fiabilité",
        [
            "Structured Output : Utilisation d'instructor + Pydantic (validation JSON).",
            "Sécurité : Protection contre les recommandations médicamenteuses.",
            "Performance : Instrumentation par composant (vLLM, Presidio, Audit).",
            "Robustesse : Fallback en cas d'échec de parsing JSON.",
        ],
    )

    show_slide(
        "Résultats",
        [
            "Latence p95 : < 1 seconde sur les échanges standards.",
            "Précision : Alignement réussi sur les protocoles de triage du CHSA.",
            "Démonstration : Déploiement Cloud opérationnel (Cloud Run + Streamlit).",
        ],
    )

    print("\n" + "=" * 70 + "\n")
    print(
        "Prêt pour la démonstration (POC) et la discussion technique (Docteur Dubois)."
    )


if __name__ == "__main__":
    main()
