# CrisisLens-AgentBench v0.1

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24429734.svg)](https://doi.org/10.5281/zenodo.24429734)

An open-source research prototype for organising public humanitarian reporting into civilian-centred categories while preserving source provenance and verbatim evidence.

**Scope:** humanitarian NLP, evidence management, uncertainty communication, and human-reviewed research. This is not a weapon, targeting, military-intelligence, surveillance, or autonomous aid-allocation system. Classification is not fact verification.

## v0.1: dataset audit, data governance, reproducible baseline and evaluation

The research dataset is **HumSet** (Fekih et al., Findings of EMNLP 2022; Apache-2.0): 154,104 expert-annotated excerpts (version 2.0.0: 129,268 train, 11,940 validation, 12,896 test) from humanitarian response documents in **English, French and Spanish**, labelled with five multi-label sets of the DEEP analytical framework (sectors, first- and second-level pillars and sub-pillars). It is downloaded from the Hugging Face Hub at run time and never redistributed; the download writes a manifest with the dataset revision and the SHA-256 of every file.

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

## v0.1 results (Kaggle, CPU)

Files: `results/kaggle-2026-11-21/` (audit, baselines, data manifest with file hashes). HumSet 2.0.0, Hub revision
`0dcf831`.

**The official split shares documents across splits.** 97.1% of validation and 96.9% of test excerpts come from a
document that also has excerpts in training (identical excerpt text: 2.7% and 2.9%). The official split is therefore
an excerpt-level split, and a model can learn a document's wording and topics from its training excerpts. We add a
**document-grouped re-split** (80/10/10 of documents, shuffled within each document's main language) in which no
document appears in two splits.

TF-IDF (word 1-2 and character 3-5 n-grams) with one-vs-rest logistic regression, per-label thresholds chosen on
validation; micro-F1 with 95% intervals from resampling whole documents:

| Task (classes) | Official split | Document-grouped split | Drop | Spanish vs English (grouped) |
|---|---|---|---|---|
| sectors (11) | 0.737 [0.729, 0.745] | 0.716 [0.702, 0.727] | −0.021 | 0.677 vs 0.725 |
| pillars_1d (7) | 0.691 [0.680, 0.702] | 0.617 [0.600, 0.635] | −0.074 | 0.500 vs 0.650 |
| pillars_2d (6) | 0.696 [0.686, 0.705] | 0.655 [0.638, 0.672] | −0.041 | 0.635 vs 0.669 |
| subpillars_1d (36) | 0.660 [0.648, 0.671] | 0.569 [0.552, 0.588] | −0.091 | 0.446 vs 0.598 |
| subpillars_2d (19) | 0.613 [0.603, 0.623] | 0.533 [0.516, 0.549] | −0.080 | 0.489 vs 0.561 |

What v0.1 shows:
- Document leakage inflates scores on the official split by 0.02 to 0.09 micro-F1, most on the fine-grained
  sub-pillars. Results on the official split overstate how a model does on a new document.
- Spanish is the hardest language on every task in the grouped split, although it has more training excerpts
  (27,810) than French (24,990); French is close to English.
- The data hold 4,210 duplicate texts within training and 774 duplicate texts with different sector labels, a floor on
  the label noise any model faces.
- Macro-F1 (in `baselines.md`) is well below micro-F1 for the sub-pillar tasks, so rare labels are where v0.2 has the
  most to gain.

v0.2 trains multilingual encoders (XLM-R) on the document-grouped split and compares them with these baselines.

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
| **v0.1** | Dataset audit, data governance, reproducible baseline and evaluation (done; results above) |
| v0.2 | Neural-network training and multilingual classification (XLM-R and multilingual encoders, seeds, calibration, cross-lingual transfer) |
| v0.3 | Multi-agent architecture: evidence retrieval and claim verification with provenance |
| v0.4 | Research benchmark: ablations, cross-language testing and robustness |
| v1.0 | Hugging Face model and dataset-card release, interactive demo, research paper and GitHub release |

Code: MIT. Dataset licences remain separate; check terms before use.

## How to cite

```bibtex
@software{hussain2026crisislens,
  author    = {Hussain, Muzammal},
  title     = {{CrisisLens-AgentBench}: dataset audit and document-grouped baselines for multilingual
               humanitarian text classification ({HumSet})},
  year      = {2026},
  version   = {0.1.2},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24429735},
  url       = {https://doi.org/10.5281/zenodo.24429735}
}
```

Please also cite HumSet: Fekih et al. (2022), *Findings of EMNLP 2022*.
