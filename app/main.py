from fastapi import FastAPI

from app.schemas import AskRequest, AskResponse, MLPrediction, SourceItem
from app.feature_extractor import extract_features
from app.ml_predictor import load_model_and_features, predict_priority
from app.retriever import retrieve_similar_tickets
from app.llm_service import (
    generate_rag_answer,
    generate_non_rag_answer,
    predict_priority_zero_shot,
)

app = FastAPI(
    title="Decision Intelligence Assistant API",
    description="Backend for ML priority prediction, RAG retrieval, and LLM comparison",
    version="1.0.0"
)

# Load model and feature names once at startup
model, feature_names = load_model_and_features()


@app.get("/")
def root():
    return {"message": "Decision Intelligence Assistant backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    query = request.query

    # 1. Retrieve similar tickets
    sources_raw = retrieve_similar_tickets(query)

    # 2. Generate answers
    rag_answer = generate_rag_answer(query, sources_raw)
    non_rag_answer = generate_non_rag_answer(query)

    # 3. Extract features and run ML model
    features = extract_features(query)
    ml_label, ml_confidence = predict_priority(model, feature_names, features)

    # Map numeric label to readable label
    label_map = {"1": "urgent", "0": "normal"}
    ml_label = label_map.get(ml_label, ml_label)

    # 4. Run LLM zero-shot priority prediction
    llm_label, llm_confidence = predict_priority_zero_shot(query)

    # 5. Convert sources into SourceItem schema objects
    sources = [
        SourceItem(
            tweet_id=source["tweet_id"],
            text=source["text"],
            score=source["score"]
        )
        for source in sources_raw
    ]

    return AskResponse(
        query=query,
        rag_answer=rag_answer,
        non_rag_answer=non_rag_answer,
        ml_prediction=MLPrediction(
            label=ml_label,
            confidence=ml_confidence
        ),
        llm_zero_shot_prediction=MLPrediction(
            label=llm_label,
            confidence=llm_confidence
        ),
        sources=sources
    )