# CrisisLens-AgentBench v0.1.0

An open-source research prototype for organising public humanitarian reporting into civilian-centred categories while preserving source provenance and verbatim evidence.

**Scope:** humanitarian NLP, evidence management, uncertainty communication, and human-reviewed research. This is not a weapon, targeting, military-intelligence, surveillance, or autonomous aid-allocation system. Classification is not fact verification.

## Included
- CPU-friendly TF-IDF + logistic-regression baseline.
- Multi-agent-style pipeline: intake, classification, evidence extraction, safety review, report assembly.
- Fictional synthetic smoke-test data only.
- CLI, tests, research protocol, data governance, and model card.

## macOS / VS Code quick start

Python 3.9+:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
crisislens train --data data/synthetic_smoke.csv --model artifacts/crisislens_baseline.joblib
crisislens analyse --input data/incoming_reports.example.jsonl --model artifacts/crisislens_baseline.joblib --output reports/example_analysis.json
```

Incoming JSONL: one JSON object per line with `report_id`, `source_name`, `source_url`, `language`, and `text`.

## Critical limitations
1. Bundled data is fictional and tests software execution only.
2. Synthetic metrics are not evidence of real-world accuracy.
3. A predicted category is not proof that an event occurred.
4. Evidence sentences are verbatim excerpts from supplied text, not independently corroborated facts.
5. v0.1.0 is English-oriented, single-label, and not validated for operational decisions.
6. Human review is mandatory before publication or use.
7. Do not ingest personal identifiers or sensitive victim information.

## Roadmap
- v0.1.0: reproducible baseline and evidence-preserving pipeline.
- v0.2.0: licensed multilingual dataset adapter, group-aware evaluation, calibration, error analysis.
- v0.3.0: source comparison and explicit contradiction detection.
- v0.4.0: multilingual evaluation and human annotation study.
- v1.0.0: model/dataset cards, reproducible paper, and demo after validation.

Code: MIT. Dataset licences remain separate; check terms before use.
