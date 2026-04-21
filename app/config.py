import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"

load_dotenv(BASE_DIR / ".env")


MODEL_PATH = ARTIFACTS_DIR / "priority_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "ml_features.json"
RAG_DATA_PATH = ARTIFACTS_DIR / "rag_subset.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT = os.getenv("VECTOR_DB_PORT", "6333")
TOP_K = int(os.getenv("TOP_K", "3"))