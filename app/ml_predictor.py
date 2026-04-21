import json
from pathlib import Path

import joblib
import pandas as pd


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "priority_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "ml_features.json"


def load_model_and_features():
    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    return model, feature_names


def predict_priority(model, feature_names, feature_dict: dict):
    row = {feature: feature_dict.get(feature, 0) for feature in feature_names}
    X = pd.DataFrame([row])

    prediction = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X)[0].max())
    else:
        confidence = 1.0

    return str(prediction), confidence