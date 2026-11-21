# Changelog

## 0.1.2 — 2026-11-21
- v0.1 results of the Kaggle run (`results/kaggle-2026-11-21/`) in the README: official split shares 97% of documents across splits; document-grouped scores 0.02 to 0.09 micro-F1 lower; Spanish hardest.
- Removed a FracAtlas notebook that Kaggle had synced into this repository.
- Fix: the HumSet loader now uses the official 2.0.0 files (`train_1_1`, `validation_1_1`, `test_1_1`). The name matcher had picked `test_gender.jsonl`, a HumSet-Bias file with 4,869 English rows and different label names, as the test split; results of that run are discarded. Only the three split files are downloaded.
- Finished tasks are reused only when they were computed on the same data files.

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
