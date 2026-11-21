"""Dataset audit for HumSet: sizes, languages, label coverage, duplicates and document leakage across splits."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict

import pandas as pd

from .humset import SPLITS, TASKS


def norm_text(text: str) -> str:
    """Lower case, no accents or punctuation, single spaces: catches copies that differ only in formatting."""
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _overlap(df: pd.DataFrame, column: str, a: str, b: str) -> Dict[str, float]:
    left = set(df.loc[df["split"] == a, column])
    rows_b = df[df["split"] == b]
    shared = rows_b[column].isin(left)
    return {"values_shared": int(len(set(rows_b[column]) & left)), "rows_in_b": int(shared.sum()),
            "share_of_b_rows": round(float(shared.mean()), 4)}


def audit(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["norm"] = df["excerpt"].map(norm_text)
    df["n_words"] = df["excerpt"].str.split().str.len()
    report: dict = {"rows": int(len(df)), "document_id_column": df.attrs.get("document_column", "lead_id")}
    report["by_split_language"] = {s: df[df["split"] == s]["lang"].value_counts().to_dict() for s in SPLITS}
    report["documents"] = {s: int(df.loc[df["split"] == s, "lead_id"].nunique()) for s in SPLITS}
    report["projects"] = {s: int(df.loc[df["split"] == s, "project_id"].nunique()) for s in SPLITS}
    report["words_per_excerpt"] = {s: df.loc[df["split"] == s, "n_words"].describe(
        percentiles=[0.5, 0.95]).round(1).to_dict() for s in SPLITS}

    labels = {}
    for task in TASKS:
        counts = df.explode(task).dropna(subset=[task]).groupby(["split", task]).size().unstack(0).fillna(0)
        n = df.groupby("split").size()
        labels[task] = {
            "classes": int(len(counts)),
            "empty_share": {s: round(float((df.loc[df["split"] == s, task].str.len() == 0).mean()), 4) for s in SPLITS},
            "labels_per_entry": {s: round(float(df.loc[df["split"] == s, task].str.len().mean()), 3) for s in SPLITS},
            "prevalence_train": (counts["train"] / n["train"]).round(4).sort_values(ascending=False).to_dict()
            if "train" in counts else {},
            "test_classes_below_20": [str(c) for c, v in counts.get("test", pd.Series(dtype=float)).items() if v < 20],
        }
        by_lang = df[df["split"] == "train"].explode(task).dropna(subset=[task]).groupby(["lang", task]).size()
        lang_n = df[df["split"] == "train"].groupby("lang").size()
        labels[task]["prevalence_train_by_language"] = (by_lang / lang_n.reindex(by_lang.index.get_level_values(0)).values
                                                        ).round(4).unstack(0).fillna(0).to_dict()
    report["labels"] = labels

    leak = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        leak[f"{a}->{b}"] = {"documents": _overlap(df, "lead_id", a, b), "projects": _overlap(df, "project_id", a, b),
                             "excerpt_text": _overlap(df, "norm", a, b)}
    report["cross_split_overlap"] = leak
    dup = df.duplicated("norm", keep=False)
    report["duplicates_within_split"] = {s: int((dup & (df["split"] == s)).sum()) for s in SPLITS}
    conflict = (df[dup].assign(key=lambda d: d[TASKS[0]].map(lambda v: tuple(sorted(v))))
                .groupby("norm")["key"].nunique())
    report["duplicate_texts_with_different_sector_labels"] = int((conflict > 1).sum())
    return report


def to_markdown(rep: dict) -> str:
    lines = ["# HumSet audit", "", f"Rows: {rep['rows']:,}", "", "| Split | Entries | Documents | Projects | Languages |",
             "|---|---|---|---|---|"]
    for s in SPLITS:
        langs = ", ".join(f"{k} {v:,}" for k, v in rep["by_split_language"][s].items())
        n = sum(rep["by_split_language"][s].values())
        lines.append(f"| {s} | {n:,} | {rep['documents'][s]:,} | {rep['projects'][s]:,} | {langs} |")
    lines += ["", "## Cross-split overlap (share of rows in the later split)", "",
              "| Pair | Same document | Same project | Same excerpt text |", "|---|---|---|---|"]
    for pair, v in rep["cross_split_overlap"].items():
        lines.append(f"| {pair} | {v['documents']['share_of_b_rows']:.1%} | {v['projects']['share_of_b_rows']:.1%} | "
                     f"{v['excerpt_text']['share_of_b_rows']:.1%} |")
    lines += ["", "## Labels", "", "| Task | Classes | Labels per entry (train) | Entries with no label (train) |",
              "|---|---|---|---|"]
    for task, v in rep["labels"].items():
        lines.append(f"| {task} | {v['classes']} | {v['labels_per_entry']['train']} | {v['empty_share']['train']:.1%} |")
    lines += ["", f"Duplicate texts within splits: {rep['duplicates_within_split']}; duplicate texts with different "
              f"sector labels: {rep['duplicate_texts_with_different_sector_labels']}.", ""]
    return "\n".join(lines)
