# Data Governance

## v0.1.0 data
The bundled CSV and JSONL contain invented fictional examples only. They do not describe real conflicts, people, organisations, or incidents.

## Admission checklist for real datasets
Record title, publisher, version, retrieval date, stable URL, licence, redistribution/training permissions, annotation process, label definitions, language and geographic coverage, sampling bias, privacy risks, and redistribution conditions. Public availability does not imply unrestricted reuse.

## Data minimisation
Do not ingest names, phone numbers, exact addresses, identity documents, or unnecessary personal data. Redact identifiers. Do not create person-level vulnerability profiles.

## Provenance and review
Retain report ID, source name, source URL, language, and verbatim evidence. A qualified human reviewer must check source credibility, context, translation, dates, geography, and potential harm before publication or operational use.

## Admitted dataset: HumSet (v0.1)

| Field | Record |
|---|---|
| Title | HumSet: Dataset of Multilingual Information Extraction and Classification for Humanitarian Crisis Response |
| Publisher | DEEP (Data Entry and Exploration Platform) NLP team |
| Source | Hugging Face Hub `nlp-thedeep/humset`; revision recorded in `manifest.json` at download |
| Licence | Apache-2.0, as stated on the dataset card and in the GitHub repository |
| Content | Excerpts selected by humanitarian analysts from public response documents, 2018 to 2021, English, French and Spanish |
| Annotation | Expert annotators using the DEEP analytical framework; multi-label |
| Redistribution here | None: the data are downloaded at run time; only aggregate statistics, scores and model outputs keyed by `entry_id` are published |
| Personal data | Excerpts come from public reports; v0.1 does not extract or store names or locations of individuals |
| Known risks | Sampling follows the projects that used DEEP; language and crisis coverage is uneven; labels reflect one framework |
