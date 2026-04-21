from openai import OpenAI

from app.config import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


def build_rag_context(sources: list) -> str:
    if not sources:
        return "No relevant sources found."

    context_parts = []
    for i, source in enumerate(sources, start=1):
        context_parts.append(
            f"Source {i}:\n"
            f"Tweet ID: {source['tweet_id']}\n"
            f"Text: {source['text']}\n"
            f"Similarity Score: {source['score']:.4f}"
        )

    return "\n\n".join(context_parts)


def generate_rag_answer(query: str, sources: list) -> str:
    context = build_rag_context(sources)

    prompt = f"""
You are a customer support knowledge assistant.

Answer the user's question using the retrieved support ticket context below.
If the context is weak or not directly relevant, say so clearly.
Do not invent facts that are not supported by the sources.

User Query:
{query}

Retrieved Context:
{context}

Answer:
""".strip()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text.strip()


def generate_non_rag_answer(query: str) -> str:
    prompt = f"""
You are a customer support assistant.

Answer the user's question as helpfully as possible without using any retrieved context.
If you are not sure, answer cautiously and avoid making unsupported claims.

User Query:
{query}

Answer:
""".strip()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text.strip()


def predict_priority_zero_shot(query: str):
    prompt = f"""
You are classifying customer support tickets.

Your task is to decide whether the following ticket is:
- urgent
- normal

Return your answer in exactly this format:
label: <urgent or normal>
confidence: <number between 0 and 1>

Ticket:
{query}
""".strip()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    text = response.output_text.strip()

    label = "normal"
    confidence = 0.5

    for line in text.splitlines():
        lower_line = line.lower().strip()
        if lower_line.startswith("label:"):
            label = line.split(":", 1)[1].strip().lower()
        elif lower_line.startswith("confidence:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 0.5

    return label, confidence