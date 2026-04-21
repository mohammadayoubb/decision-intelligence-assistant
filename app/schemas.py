from typing import List
from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str


class MLPrediction(BaseModel):
    label: str
    confidence: float


class SourceItem(BaseModel):
    tweet_id: str
    text: str
    score: float


class AskResponse(BaseModel):
    query: str
    rag_answer: str
    non_rag_answer: str
    ml_prediction: MLPrediction
    llm_zero_shot_prediction: MLPrediction
    sources: List[SourceItem]