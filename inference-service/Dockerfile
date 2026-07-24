FROM vllm/vllm-openai:latest

# Installer FastAPI et Gradio
RUN pip install fastapi gradio

# Copier le script d'orchestration
COPY start_orchestrated.sh .
RUN chmod +x start_orchestrated.sh

# Exposer les ports
EXPOSE 8000
EXPOSE 7860
EXPOSE 8080

# Démarrage
ENTRYPOINT []
CMD ["./start_orchestrated.sh"]
