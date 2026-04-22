import os
from qdrant_client import QdrantClient

COLLECTION_NAME = "support_tickets"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
TOP_K = 3

client = QdrantClient(url=QDRANT_URL)


def retrieve_similar_tickets(query: str, top_k: int = TOP_K):
    results = client.query(
        collection_name=COLLECTION_NAME,
        query_text=query,
        limit=top_k,
    )

    formatted_results = []

    for result in results:
        payload = result.metadata if hasattr(result, "metadata") else result.payload

        formatted_results.append({
            "tweet_id": str(payload.get("tweet_id", "")),
            "text": payload.get("text", ""),
            "score": float(result.score),
        })

    return formatted_results