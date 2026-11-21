from crisislens_agentbench.data import load_training_csv, load_reports_jsonl

def test_load_synthetic_training_data():
    frame = load_training_csv("data/synthetic_smoke.csv")
    assert len(frame) >= 40
    assert frame["label"].nunique() == 8

def test_load_example_reports():
    reports = load_reports_jsonl("data/incoming_reports.example.jsonl")
    assert len(reports) == 2
    assert reports[0].report_id == "DEMO-001"
