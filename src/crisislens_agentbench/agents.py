import re
from .schemas import Finding

class IntakeAgent:
    def process(self, report):
        if len(report.text.strip()) < 8:
            raise ValueError("Report text is too short for meaningful analysis.")
        return report

class ClassificationAgent:
    def __init__(self, model):
        self.model = model

    def process(self, report):
        probabilities = self.model.predict_proba([report.text])[0]
        ranked = sorted(zip(self.model.classes_, probabilities),
                        key=lambda item: item[1], reverse=True)
        label, score = ranked[0]
        scores = [{"category": str(cat), "score": float(prob)}
                  for cat, prob in ranked]
        return label, float(score), scores

class EvidenceAgent:
    def process(self, text):
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text)
                     if part.strip()]
        return sentences[:5]

class SafetyReviewAgent:
    def process(self, confidence, language, evidence):
        warnings = []
        if confidence < 0.55:
            warnings.append("Low model confidence: manual review required.")
        if language.lower() not in {"en", "eng", "english"}:
            warnings.append("Non-English input: this v0.1.0 model is English-only.")
        if not evidence:
            warnings.append("No extractable evidence sentence was found.")
        warnings += [
            "Classification is not fact-checking or independent corroboration.",
            "Human review is required before reuse or publication.",
        ]
        return True, warnings

class ReportAssemblyAgent:
    def process(self, report, classification, evidence, safety):
        label, confidence, scores = classification
        review_required, warnings = safety
        finding = Finding(
            report_id=report.report_id, category=label, confidence=confidence,
            evidence=evidence, source_name=report.source_name,
            source_url=report.source_url, language=report.language,
            review_required=review_required, warnings=warnings,
        )
        result = finding.to_dict()
        result["category_scores"] = scores
        return result
