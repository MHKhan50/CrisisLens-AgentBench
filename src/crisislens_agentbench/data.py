import json
from pathlib import Path
from typing import List
import pandas as pd
from .schemas import Report

REQUIRED_TRAIN_COLUMNS = {"text", "label"}
ALLOWED_LABELS = {"displacement", "food_security", "health", "shelter",
                  "education", "infrastructure", "protection", "other"}

def load_training_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_TRAIN_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("Training CSV missing columns: " + ", ".join(sorted(missing)))
    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["label"] != "")]
    unknown = sorted(set(frame["label"]) - ALLOWED_LABELS)
    if unknown:
        raise ValueError("Unsupported labels: " + ", ".join(unknown))
    if frame["label"].nunique() < 2:
        raise ValueError("At least two distinct labels are required for training.")
    return frame.reset_index(drop=True)

def load_reports_jsonl(path: str) -> List[Report]:
    reports = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                reports.append(Report.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError("Invalid JSONL record at line {}: {}".format(line_number, exc))
    if not reports:
        raise ValueError("No reports found in input JSONL.")
    return reports
