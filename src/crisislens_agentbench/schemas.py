from dataclasses import dataclass, asdict
from typing import Any, Dict, List

@dataclass
class Report:
    report_id: str
    source_name: str
    source_url: str
    language: str
    text: str

    @classmethod
    def from_dict(cls, item: Dict[str, Any]) -> "Report":
        required = ("report_id", "source_name", "source_url", "language", "text")
        missing = [key for key in required if not str(item.get(key, "")).strip()]
        if missing:
            raise ValueError("Missing required report fields: " + ", ".join(missing))
        return cls(**{key: str(item[key]) for key in required})

@dataclass
class Finding:
    report_id: str
    category: str
    confidence: float
    evidence: List[str]
    source_name: str
    source_url: str
    language: str
    review_required: bool
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
