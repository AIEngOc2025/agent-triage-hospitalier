from vllm import LLM

llm = LLM(
    model="models/merged_dpo_final_chsa",
)
conversation = [
    {
        "role": "system",
        "content": "Tu es l'infirmier de triage du CHSA. Pose une question courte à la fois pour évaluer l'urgence. Sois bilingue.",
    },
    {
        "role": "user",
        "content": "Hello",
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?",
    },
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]
outputs = llm.chat(conversation)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
