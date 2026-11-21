
cat >> src/crisislens_agentbench/modeling.py <<'PY'


def benchmark_model(frame, test_size: float = 0.25, random_state: int = 17):
    """Train on a stratified partition and evaluate on held-out data."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")

    label_counts = frame["label"].value_counts()
    if label_counts.min() < 2:
        raise ValueError(
            "Every label needs at least two examples for stratified splitting."
        )

    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame["label"],
    )

    model = build_model(random_state=random_state)
    model.fit(train_frame["text"].tolist(), train_frame["label"].tolist())
    predictions = model.predict(test_frame["text"].tolist())

    return {
        "evaluation_type": "stratified_holdout",
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_frame)),
        "rows_test": int(len(test_frame)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "accuracy": float(accuracy_score(test_frame["label"], predictions)),
        "macro_f1": float(
            f1_score(test_frame["label"], predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(test_frame["label"], predictions, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            test_frame["label"], predictions, output_dict=True, zero_division=0
        ),
        "warning": (
            "Exploratory evaluation on a small synthetic dataset. "
            "These results are not evidence of real-world humanitarian "
            "report classification performance."
        ),
    }
PY
