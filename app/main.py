import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import (
    AskRequest,
    AskResponse,
    PredictionResult,
    SourceItem,
    CostBreakdown,
)
from app.feature_extractor import extract_features
from app.ml_predictor import load_model_and_features, predict_priority
from app.retriever import retrieve_similar_tickets
from app.llm_service import (
    generate_rag_answer,
    generate_non_rag_answer,
    predict_priority_zero_shot,
)
from app.logging_utils import log_query_result

app = FastAPI(
    title="Decision Intelligence Assistant API",
    description="Backend for ML priority prediction, RAG retrieval, and LLM comparison",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model, feature_names = load_model_and_features()


@app.get("/")
def root():
    return {"message": "Decision Intelligence Assistant backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    total_start = time.perf_counter()
    query = request.query

    # 1) Retrieval
    retrieval_start = time.perf_counter()
    sources_raw = retrieve_similar_tickets(query)
    retrieval_latency_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

    # 2) LLM answers
    rag_result = generate_rag_answer(query, sources_raw)
    non_rag_result = generate_non_rag_answer(query)

    # 3) ML prediction
    ml_start = time.perf_counter()
    features = extract_features(query)
    ml_label, ml_confidence = predict_priority(model, feature_names, features)
    ml_latency_ms = round((time.perf_counter() - ml_start) * 1000, 2)

    # Map numeric ML labels to readable labels
    label_map = {"1": "urgent", "0": "normal"}
    ml_label = label_map.get(ml_label, ml_label)

    # 4) LLM zero-shot prediction
    llm_result = predict_priority_zero_shot(query)

    # 5) Convert sources to schema objects
    sources = [
        SourceItem(
            tweet_id=source["tweet_id"],
            text=source["text"],
            score=source["score"]
        )
        for source in sources_raw
    ]

    # 6) Cost breakdown
    rag_cost = rag_result["cost_usd"]
    non_rag_cost = non_rag_result["cost_usd"]
    zero_shot_cost = llm_result["cost_usd"]
    total_llm_cost = round(rag_cost + non_rag_cost + zero_shot_cost, 6)

    # 7) Total latency
    total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)

    response = AskResponse(
        query=query,
        rag_answer=rag_result["answer"],
        non_rag_answer=non_rag_result["answer"],
        ml_prediction=PredictionResult(
            label=ml_label,
            confidence=ml_confidence,
            latency_ms=ml_latency_ms
        ),
        llm_zero_shot_prediction=PredictionResult(
            label=llm_result["label"],
            confidence=llm_result["confidence"],
            latency_ms=llm_result["latency_ms"]
        ),
        sources=sources,
        rag_latency_ms=rag_result["latency_ms"],
        non_rag_latency_ms=non_rag_result["latency_ms"],
        retrieval_latency_ms=retrieval_latency_ms,
        total_latency_ms=total_latency_ms,
        cost=CostBreakdown(
            rag_cost_usd=rag_cost,
            non_rag_cost_usd=non_rag_cost,
            zero_shot_cost_usd=zero_shot_cost,
            total_llm_cost_usd=total_llm_cost
        )
    )

    # 8) Logging
    log_query_result({
        "query": query,
        "sources": sources_raw,
        "rag_answer": rag_result["answer"],
        "non_rag_answer": non_rag_result["answer"],
        "ml_prediction": {
            "label": ml_label,
            "confidence": ml_confidence,
            "latency_ms": ml_latency_ms
        },
        "llm_zero_shot_prediction": llm_result,
        "retrieval_latency_ms": retrieval_latency_ms,
        "rag_latency_ms": rag_result["latency_ms"],
        "non_rag_latency_ms": non_rag_result["latency_ms"],
        "total_latency_ms": total_latency_ms,
        "cost": {
            "rag_cost_usd": rag_cost,
            "non_rag_cost_usd": non_rag_cost,
            "zero_shot_cost_usd": zero_shot_cost,
            "total_llm_cost_usd": total_llm_cost
        }
    })

    return response