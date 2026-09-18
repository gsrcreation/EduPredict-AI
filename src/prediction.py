from pathlib import Path
import pandas as pd
import joblib
from .data_preprocessing import FEATURES, validate_student_input

MODEL_DIR = Path("models")

def load_models():
    reg = joblib.load(MODEL_DIR/"performance_model.pkl")
    cls = joblib.load(MODEL_DIR/"risk_model.pkl")
    return reg, cls

def predict_student(data):
    validate_student_input(data)
    X = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    reg, cls = load_models()
    score = float(reg.predict(X)[0])
    risk = str(cls.predict(X)[0])
    if hasattr(cls, "predict_proba"):
        confidence = float(cls.predict_proba(X).max())
    else:
        confidence = None
    return {
        "predicted_score": round(max(0, min(100, score)), 2),
        "risk_level": risk,
        "confidence": None if confidence is None else round(confidence, 3)
    }