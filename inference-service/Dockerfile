FROM vllm/vllm-openai:latest

# Installer FastAPI et Gradio
RUN pip install fastapi gradio

# Exposer le port utilisé par vLLM et Gradio
EXPOSE 8000
EXPOSE 7860

# Démarrage vLLM
CMD ["--model", "Qwen/Qwen3-1.7B-Base", "--host", "0.0.0.0", "--port", "8000"]
