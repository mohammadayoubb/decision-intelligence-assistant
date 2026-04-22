import os
from pathlib import Path

# Set cache path BEFORE importing qdrant_client
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "fastembed_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["FASTEMBED_CACHE_PATH"] = str(CACHE_DIR)

import pandas as pd
from qdrant_client import QdrantClient

COLLECTION_NAME = "support_tickets"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
RAG_CSV_PATH = BASE_DIR / "artifacts" / "rag_subset.csv"
EMBEDDING_MODEL = "BAAI/bge-small-en"


def load_rag_subset(csv_path=RAG_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)

    if "tweet_id" not in df.columns:
        raise ValueError("rag_subset.csv must contain a 'tweet_id' column")

    return df


def build_metadata(df: pd.DataFrame):
    metadata = []
    for _, row in df.iterrows():
        item = {
            "tweet_id": str(row["tweet_id"]),
            "text": row["text"],
        }

        if "priority_label" in df.columns:
            item["priority_label"] = str(row["priority_label"])

        metadata.append(item)

    return metadata


def load_into_qdrant():
    df = load_rag_subset()

    client = QdrantClient(url=QDRANT_URL)

    client.set_model(EMBEDDING_MODEL, cache_dir=str(CACHE_DIR))

    documents = df["text"].tolist()
    metadata = build_metadata(df)
    ids = list(range(len(df)))

    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]

    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    client.add(
        collection_name=COLLECTION_NAME,
        documents=documents,
        metadata=metadata,
        ids=ids,
    )

    print(f"Loaded {len(documents)} documents into Qdrant collection '{COLLECTION_NAME}'")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"FastEmbed cache dir: {CACHE_DIR}")
    print(f"Qdrant URL: {QDRANT_URL}")


if __name__ == "__main__":
    load_into_qdrant()