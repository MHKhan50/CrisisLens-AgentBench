# CrisisLens-AgentBench v0.1

An open-source research prototype for organising public humanitarian reporting into civilian-centred categories while preserving source provenance and verbatim evidence.

**Scope:** humanitarian NLP, evidence management, uncertainty communication, and human-reviewed research. This is not a weapon, targeting, military-intelligence, surveillance, or autonomous aid-allocation system. Classification is not fact verification.

## v0.1: dataset audit, data governance, reproducible baseline and evaluation

The research dataset is **HumSet** (Fekih et al., Findings of EMNLP 2022; Apache-2.0): about 149,000 expert-annotated excerpts from humanitarian response documents in **English, French and Spanish**, labelled with five multi-label sets of the DEEP analytical framework (sectors, first- and second-level pillars and sub-pillars). It is downloaded from the Hugging Face Hub at run time and never redistributed; the download writes a manifest with the dataset revision and the SHA-256 of every file.

v0.1 answers three questions before any neural model is trained:

1. **What is in the data?** Sizes per split and language, documents and projects, label counts, labels per excerpt, excerpts with no label, prevalence by language.
2. **Can the split be trusted?** Share of validation and test excerpts whose source document, project or normalised text also occurs in training, and duplicate texts with conflicting labels.
3. **How far does a transparent baseline get?** A prior baseline and TF-IDF word and character n-grams with one-vs-rest logistic regression on all five tasks, thresholds chosen on validation only, micro- and macro-F1 with 95% intervals that resample whole documents, scores per language, on the official split and on a document-grouped re-split where no document appears in two splits.

Run on Kaggle (CPU, Internet on): notebook `notebooks/kaggle_humset_v01.ipynb`, Kaggle name **"CrisisLens HumSet Audit and Baselines (CPU)"**. Or locally:

```bash
python -m pip install -e ".[dev]"
pytest -q
crisislens humset-download --data-dir data/humset
crisislens humset-audit --data-dir data/humset --reports reports
crisislens humset-baselines --data-dir data/humset --reports reports
```

The v0.1.0 scaffold is kept: an evidence-preserving pipeline (intake, classification, evidence extraction, safety review, report assembly) on fictional smoke-test data, `crisislens train / evaluate / analyse`.

## Critical limitations
1. Bundled data is fictional and tests software execution only.
2. Synthetic metrics are not evidence of real-world accuracy.
3. A predicted category is not proof that an event occurred.
4. Evidence sentences are verbatim excerpts from supplied text, not independently corroborated facts.
5. v0.1.0 is English-oriented, single-label, and not validated for operational decisions.
6. Human review is mandatory before publication or use.
7. Do not ingest personal identifiers or sensitive victim information.

## Roadmap

| Version | Scope |
|---|---|
| **v0.1** | Dataset audit, data governance, reproducible baseline and evaluation (this release) |
| v0.2 | Neural-network training and multilingual classification (XLM-R and multilingual encoders, seeds, calibration, cross-lingual transfer) |
| v0.3 | Multi-agent architecture: evidence retrieval and claim verification with provenance |
| v0.4 | Research benchmark: ablations, cross-language testing and robustness |
| v1.0 | Hugging Face model and dataset-card release, interactive demo, research paper and GitHub release |

Code: MIT. Dataset licences remain separate; check terms before use.
