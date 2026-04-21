from fastapi import FastAPI
from app.schemas import AskRequest, AskResponse, MLPrediction
from app.feature_extractor import extract_features
from app.ml_predictor import load_model_and_features, predict_priority

app = FastAPI(
    title="Decision Intelligence Assistant API",
    description="Backend for ML priority prediction, RAG retrieval, and LLM comparison",
    version="1.0.0"
)

# Load model and feature names once when the app starts
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

    # Extract engineered features from the user query
    features = extract_features(query)

    # Predict priority using the saved ML model
    ml_label, ml_confidence = predict_priority(model, feature_names, features)

    # Placeholder outputs for now
    rag_answer = "RAG answer placeholder"
    non_rag_answer = "Non-RAG answer placeholder"
    llm_zero_shot_label = "urgent"
    llm_zero_shot_confidence = 0.85
    sources = []

    return AskResponse(
        query=query,
        rag_answer=rag_answer,
        non_rag_answer=non_rag_answer,
        ml_prediction=MLPrediction(
            label=ml_label,
            confidence=ml_confidence
        ),
        llm_zero_shot_prediction=MLPrediction(
            label=llm_zero_shot_label,
            confidence=llm_zero_shot_confidence
        ),
        sources=sources
    )