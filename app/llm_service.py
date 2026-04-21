import time
from openai import OpenAI
from app.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
INPUT_COST_PER_1K = 0.00015
OUTPUT_COST_PER_1K = 0.00060


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (output_tokens / 1000) * OUTPUT_COST_PER_1K
    return round(input_cost + output_cost, 6)


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


def generate_rag_answer(query: str, sources: list):
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

    start_time = time.perf_counter()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    input_tokens = response.usage.input_tokens if response.usage else 0
    output_tokens = response.usage.output_tokens if response.usage else 0
    cost_usd = estimate_cost(input_tokens, output_tokens)

    return {
        "answer": response.output_text.strip(),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd
    }


def generate_non_rag_answer(query: str):
    prompt = f"""
You are a customer support assistant.

Answer the user's question as helpfully as possible without using any retrieved context.
If you are not sure, answer cautiously and avoid making unsupported claims.

User Query:
{query}

Answer:
""".strip()

    start_time = time.perf_counter()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    input_tokens = response.usage.input_tokens if response.usage else 0
    output_tokens = response.usage.output_tokens if response.usage else 0
    cost_usd = estimate_cost(input_tokens, output_tokens)

    return {
        "answer": response.output_text.strip(),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd
    }


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

    start_time = time.perf_counter()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    input_tokens = response.usage.input_tokens if response.usage else 0
    output_tokens = response.usage.output_tokens if response.usage else 0
    cost_usd = estimate_cost(input_tokens, output_tokens)

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

    return {
        "label": label,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd
    }