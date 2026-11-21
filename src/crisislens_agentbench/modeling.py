from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def build_model(random_state: int = 17) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000,
                                  strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=random_state)),
    ])

def train_model(frame, output_path: str, random_state: int = 17):
    model = build_model(random_state=random_state)
    model.fit(frame["text"].tolist(), frame["label"].tolist())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model

def load_model(path: str):
    return joblib.load(path)

def evaluate_model(model, frame):
    predictions = model.predict(frame["text"].tolist())
    return {
        "rows_evaluated": int(len(frame)),
        "accuracy_on_supplied_data": float(accuracy_score(frame["label"], predictions)),
        "classification_report": classification_report(
            frame["label"], predictions, output_dict=True, zero_division=0
        ),
        "warning": "If evaluated on training data, these are resubstitution metrics, not an unbiased estimate of generalisation.",
    }
