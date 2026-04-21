from typing import List
from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str


class PredictionResult(BaseModel):
    label: str
    confidence: float
    latency_ms: float


class SourceItem(BaseModel):
    tweet_id: str
    text: str
    score: float


class CostBreakdown(BaseModel):
    rag_cost_usd: float
    non_rag_cost_usd: float
    zero_shot_cost_usd: float
    total_llm_cost_usd: float


class AskResponse(BaseModel):
    query: str
    rag_answer: str
    non_rag_answer: str
    ml_prediction: PredictionResult
    llm_zero_shot_prediction: PredictionResult
    sources: List[SourceItem]
    rag_latency_ms: float
    non_rag_latency_ms: float
    retrieval_latency_ms: float
    total_latency_ms: float
    cost: CostBreakdown