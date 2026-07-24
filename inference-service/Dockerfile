FROM vllm/vllm-openai:latest

# Set environment variables for CUDA compatibility as per vLLM docs
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1

# Clean existing conflicting packages and force reinstall correct versions
RUN pip uninstall -y transformers torch && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.4.0 transformers[torch]==5.14.1

# Installer FastAPI et Gradio
RUN pip install --no-cache-dir fastapi gradio

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
