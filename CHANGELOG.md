# Changelog

## 0.1.1 — 2026-11-21
- HumSet adapter: download from the Hugging Face Hub with a provenance manifest (revision, file hashes); labels normalised to lists.
- Dataset audit: sizes, languages, documents, label coverage, duplicates and cross-split overlap of documents, projects and text.
- Multi-label baselines on all five HumSet tasks (prior; TF-IDF word and character n-grams with logistic regression), validation-chosen thresholds, document-level bootstrap intervals, per-language scores, official and document-grouped splits.
- Kaggle notebook, CI on Python 3.9 and 3.11, data-governance record for HumSet, analysis plan; repaired the modeling test.

## 0.1.0 — 2026-11-21
- Initial research scaffold.
- CPU-friendly TF-IDF/logistic-regression baseline.
- Evidence-preserving multi-agent-style pipeline.
- Synthetic smoke-test data, tests, research protocol, data governance, and model card.
