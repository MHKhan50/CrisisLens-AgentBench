from crisislens_agentbench.data import load_training_csv
from crisislens_agentbench.modeling import evaluate_model, train_model


def test_baseline_trains_and_evaluates(tmp_path):
    frame = load_training_csv("data/synthetic_smoke.csv")
    model = train_model(frame, str(tmp_path / "model.joblib"))
    result = evaluate_model(model, frame)
    assert result["rows_evaluated"] == len(frame)
    assert 0.0 <= result["accuracy_on_supplied_data"] <= 1.0
    assert "resubstitution" in result["warning"]
