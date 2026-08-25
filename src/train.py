import os
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "complaints.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_model(texts, labels):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(texts, labels)
    return model


def evaluate(model, x_test, y_test, name):
    predictions = model.predict(x_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(classification_report(y_test, predictions, zero_division=0))


def main():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["complaint_text"].apply(clean_text)

    x_train, x_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        df["clean_text"], df["category"], df["priority"],
        test_size=0.25, random_state=42
    )

    category_model = train_model(x_train, y_cat_train)
    priority_model = train_model(x_train, y_pri_train)

    evaluate(category_model, x_test, y_cat_test, "Category Model")
    evaluate(priority_model, x_test, y_pri_test, "Priority Model")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(category_model, os.path.join(MODEL_DIR, "category_model.joblib"))
    joblib.dump(priority_model, os.path.join(MODEL_DIR, "priority_model.joblib"))
    print("\nModels saved in models/")


if __name__ == "__main__":
    main()
