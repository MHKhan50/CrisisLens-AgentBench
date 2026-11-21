"""Reproducible multi-label baselines and evaluation on HumSet.

Baselines per task: a prior baseline (each label predicted with its training prevalence) and TF-IDF word + character
n-grams with one-vs-rest logistic regression. Per-label thresholds are chosen on the validation split only.
Intervals are 95% percentile bootstrap intervals that resample whole documents (lead_id), because excerpts from one
document are correlated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from .humset import TASKS

GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def binarise(train_labels, *others):
    mlb = MultiLabelBinarizer()
    mlb.fit(list(train_labels))
    keep = [c for c in mlb.classes_]
    return mlb, [mlb.transform(list(x)) for x in (train_labels, *others)], keep


def choose_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    th = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        scores = [f1_score(y[:, j], p[:, j] >= t, zero_division=0) for t in GRID]
        th[j] = GRID[int(np.argmax(scores))]
    return th


def scores(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Micro, macro (over labels present in ``y``) and sample-averaged F1; NaN when undefined (no rows or labels)."""
    nan = float("nan")
    if y.shape[0] == 0 or y.shape[1] == 0:
        return {"micro_f1": nan, "macro_f1": nan, "samples_f1": nan}
    present = y.sum(axis=0) > 0
    return {"micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y[:, present], pred[:, present], average="macro", zero_division=0))
            if present.any() else nan,
            "samples_f1": float(f1_score(y, pred, average="samples", zero_division=0))}


def grouped_bootstrap(y, pred, groups, n_boot: int = 1000, seed: int = 0) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(groups))
    index = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    out = {k: [] for k in ("micro_f1", "macro_f1")}
    for _ in range(n_boot):
        pick = np.concatenate([index[g] for g in rng.integers(0, len(uniq), len(uniq))])
        s = scores(y[pick], pred[pick])
        for k in out:
            out[k].append(s[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in out.items()}


def tfidf_model(seed: int = 0, max_features: int = 200_000, n_jobs: int = -1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True,
                                 strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=max_features,
                                 sublinear_tf=True)),
    ])
    clf = OneVsRestClassifier(LogisticRegression(C=4.0, solver="liblinear", max_iter=1000, random_state=seed),
                              n_jobs=n_jobs)
    return Pipeline([("features", features), ("clf", clf)])


def run_task(df: pd.DataFrame, task: str, n_boot: int = 1000, seed: int = 0, model_factory=tfidf_model,
             exclude_test_rows: Optional[pd.Series] = None) -> dict:
    tr, va, te = (df[df["split"] == s] for s in ("train", "validation", "test"))
    mlb, (ytr, yva, yte), classes = binarise(tr[task], va[task], te[task])
    keep = ytr.sum(axis=0) > 0
    result = {"task": task, "classes": [str(c) for c in classes], "n": {"train": len(tr), "validation": len(va),
                                                                          "test": len(te)}}
    prior = ytr.mean(axis=0)
    p_prior_va = np.tile(prior, (len(va), 1)); p_prior_te = np.tile(prior, (len(te), 1))
    model = model_factory(seed=seed)
    model.fit(tr["excerpt"].tolist(), ytr[:, keep])
    p_va = np.zeros(yva.shape); p_te = np.zeros(yte.shape)
    p_va[:, keep] = model.predict_proba(va["excerpt"].tolist())
    p_te[:, keep] = model.predict_proba(te["excerpt"].tolist())
    mask = np.ones(len(te), bool) if exclude_test_rows is None else ~exclude_test_rows.loc[te.index].to_numpy()
    for name, pv, pt in (("prior", p_prior_va, p_prior_te), ("tfidf_lr", p_va, p_te)):
        th = choose_thresholds(yva, pv)
        pred = (pt >= th).astype(int)
        res = {"test": scores(yte, pred), "validation": scores(yva, (pv >= th).astype(int)),
               "test_ci": grouped_bootstrap(yte, pred, te["lead_id"].to_numpy(), n_boot, seed),
               "by_language": {lang: scores(yte[te["lang"].to_numpy() == lang], pred[te["lang"].to_numpy() == lang])
                               for lang in sorted(te["lang"].unique())},
               "per_label_f1": {str(c): float(f1_score(yte[:, j], pred[:, j], zero_division=0))
                                for j, c in enumerate(classes)},
               "thresholds": {str(c): float(t) for c, t in zip(classes, th)}}
        if exclude_test_rows is not None:
            res["test_without_flagged_rows"] = scores(yte[mask], pred[mask])
            res["flagged_test_rows"] = int((~mask).sum())
        result[name] = res
        if name == "tfidf_lr":
            result["_test_probabilities"] = pt
    return result


def run_all(df: pd.DataFrame, out_dir: str, tasks=TASKS, n_boot: int = 1000, seed: int = 0,
            model_factory=tfidf_model, exclude_test_rows: Optional[pd.Series] = None, tag: str = "official") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in tasks:
        done = out / f"{tag}_{task}.json"
        signature = df.attrs.get("data_signature")
        r = json.loads(done.read_text()) if done.exists() else None
        if r is not None and r.get("data_signature") == signature:  # finished earlier on the same data files
            summary[task] = {m: {"micro_f1": r[m]["test"]["micro_f1"], "macro_f1": r[m]["test"]["macro_f1"],
                                 "micro_ci": r[m]["test_ci"]["micro_f1"], "macro_ci": r[m]["test_ci"]["macro_f1"],
                                 "by_language_micro": {k: v["micro_f1"] for k, v in r[m]["by_language"].items()}}
                             for m in ("prior", "tfidf_lr")}
            print(f"{tag} {task}: reused", flush=True)
            continue
        r = run_task(df, task, n_boot, seed, model_factory, exclude_test_rows)
        probs = r.pop("_test_probabilities")
        te = df[df["split"] == "test"]
        np.savez_compressed(out / f"{tag}_{task}_tfidf_test_probs.npz", probs=probs, entry_id=te["entry_id"].to_numpy())
        r["data_signature"] = signature
        (out / f"{tag}_{task}.json").write_text(json.dumps(r, indent=2))
        summary[task] = {m: {"micro_f1": r[m]["test"]["micro_f1"], "macro_f1": r[m]["test"]["macro_f1"],
                             "micro_ci": r[m]["test_ci"]["micro_f1"], "macro_ci": r[m]["test_ci"]["macro_f1"],
                             "by_language_micro": {k: v["micro_f1"] for k, v in r[m]["by_language"].items()}}
                         for m in ("prior", "tfidf_lr")}
        print(json.dumps({task: summary[task]["tfidf_lr"]}), flush=True)
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def grouped_resplit(df: pd.DataFrame, seed: int = 0, fractions=(0.8, 0.1, 0.1)) -> pd.DataFrame:
    """Re-split by document (lead_id) so that no document appears in two splits.

    Documents are shuffled within their main language, so each split keeps roughly the original language mix.
    """
    rng = np.random.default_rng(seed)
    main_lang = df.groupby("lead_id")["lang"].agg(lambda s: s.value_counts().index[0])
    assign = {}
    for _, leads in main_lang.groupby(main_lang):
        ids = leads.index.to_numpy().copy()
        rng.shuffle(ids)
        a = int(len(ids) * fractions[0]); b = int(len(ids) * (fractions[0] + fractions[1]))
        assign.update({l: "train" for l in ids[:a]}); assign.update({l: "validation" for l in ids[a:b]})
        assign.update({l: "test" for l in ids[b:]})
    out = df.copy()
    out["split"] = out["lead_id"].map(assign)
    return out


def summary_markdown(summary: dict, title: str) -> str:
    lines = [f"## {title}", "", "| Task | Prior micro-F1 | TF-IDF LR micro-F1 [95% CI] | TF-IDF LR macro-F1 [95% CI] | "
             "micro-F1 en / fr / es |", "|---|---|---|---|---|"]
    for task, v in summary.items():
        t = v["tfidf_lr"]; langs = " / ".join(f"{t['by_language_micro'].get(l, float('nan')):.3f}" for l in ("en", "fr", "es"))
        lines.append(f"| {task} | {v['prior']['micro_f1']:.3f} | {t['micro_f1']:.3f} [{t['micro_ci'][0]:.3f}, "
                     f"{t['micro_ci'][1]:.3f}] | {t['macro_f1']:.3f} [{t['macro_ci'][0]:.3f}, {t['macro_ci'][1]:.3f}] | {langs} |")
    return "\n".join(lines) + "\n"
