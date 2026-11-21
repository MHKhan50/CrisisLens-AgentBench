from crisislens_agentbench.data import load_training_csv, load_reports_jsonl
from crisislens_agentbench.modeling import train_model
from crisislens_agentbench.pipeline import CrisisLensPipeline

def test_pipeline_preserves_provenance_and_review_flag(tmp_path):
    frame = load_training_csv("data/synthetic_smoke.csv")
    model = train_model(frame, str(tmp_path / "model.joblib"))
    report = load_reports_jsonl("data/incoming_reports.example.jsonl")[0]
    result = CrisisLensPipeline(model).analyse(report)
    assert result["report_id"] == "DEMO-001"
    assert result["source_url"] == "https://example.org/demo"
    assert result["review_required"] is True
    assert result["evidence"]
    assert any("not fact-checking" in warning for warning in result["warnings"])
