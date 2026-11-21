from .agents import (IntakeAgent, ClassificationAgent, EvidenceAgent,
                     SafetyReviewAgent, ReportAssemblyAgent)

class CrisisLensPipeline:
    def __init__(self, model):
        self.intake = IntakeAgent()
        self.classifier = ClassificationAgent(model)
        self.evidence = EvidenceAgent()
        self.safety = SafetyReviewAgent()
        self.assembler = ReportAssemblyAgent()

    def analyse(self, report):
        report = self.intake.process(report)
        classification = self.classifier.process(report)
        evidence = self.evidence.process(report.text)
        safety = self.safety.process(classification[1], report.language, evidence)
        return self.assembler.process(report, classification, evidence, safety)
