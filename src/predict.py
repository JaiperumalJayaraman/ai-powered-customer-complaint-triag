import os
import re
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_models():
    category = joblib.load(os.path.join(MODEL_DIR, "category_model.joblib"))
    priority = joblib.load(os.path.join(MODEL_DIR, "priority_model.joblib"))
    return category, priority


def predict(complaint):
    category_model, priority_model = load_models()
    text = clean_text(complaint)

    category = category_model.predict([text])[0]
    priority = priority_model.predict([text])[0]
    category_confidence = max(category_model.predict_proba([text])[0])
    priority_confidence = max(priority_model.predict_proba([text])[0])

    return {
        "category": category,
        "priority": priority,
        "category_confidence": category_confidence,
        "priority_confidence": priority_confidence,
    }
