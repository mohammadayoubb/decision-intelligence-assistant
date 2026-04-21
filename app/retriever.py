from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import RAG_DATA_PATH, TOP_K


@lru_cache(maxsize=1)
def load_rag_data():
    df = pd.read_csv(RAG_DATA_PATH)

    # Keep only rows with usable text
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)

    return df


@lru_cache(maxsize=1)
def build_vector_store():
    df = load_rag_data()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(df["text"])

    return df, vectorizer, matrix


def retrieve_similar_tickets(query: str, top_k: int = TOP_K):
    df, vectorizer, matrix = build_vector_store()

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "tweet_id": str(row["tweet_id"]),
            "text": row["text"],
            "score": float(similarities[idx])
        })

    return results