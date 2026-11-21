# Research Protocol — v0.1.0

## Research question
Can a lightweight, evidence-preserving NLP pipeline organise public humanitarian reporting into civilian-needs categories while retaining provenance and communicating uncertainty?

## Contribution
This release is a reproducible software and evaluation scaffold, not a claim of a novel neural architecture or validated operational performance. The baseline is TF-IDF + logistic regression. The agentic design separates intake, classification, evidence extraction, safety review, and report assembly.

## Unit of analysis
One supplied document. The model predicts one category: displacement, food_security, health, shelter, education, infrastructure, protection, or other. Real reports may contain multiple needs; future work should assess multi-label classification.

## Experimental controls
- Fixed seed (default 17).
- Explicit data and model paths.
- No network calls during training or inference.
- Synthetic smoke data is not suitable for real-world performance claims.
- Evaluation on training data is resubstitution evaluation only.

## v0.2.0 plan
1. Select a dataset with explicit licence and annotation documentation.
2. Preserve document IDs and group by source/document/event to prevent leakage.
3. Use held-out train/validation/test partitions.
4. Report macro-F1, per-class precision/recall/F1, confusion matrix, calibration, and confidence intervals.
5. Analyse errors by language, source type, topic, and document length.
6. Compare TF-IDF with a compact pretrained transformer after governance review.
7. Human-review evidence relevance and category validity.
8. Publish data statements, model card, limitations, and exact reproduction commands.

## Safety
For humanitarian research and civilian-support information organisation only. No target selection, combat operations, surveillance of individuals, or person-level vulnerability profiling. It does not verify events, attribute responsibility, or establish casualty figures.
