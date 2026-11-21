#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e ".[dev]"
pytest -q
crisislens train --data data/synthetic_smoke.csv --model artifacts/crisislens_baseline.joblib
crisislens analyse --input data/incoming_reports.example.jsonl --model artifacts/crisislens_baseline.joblib --output reports/example_analysis.json
