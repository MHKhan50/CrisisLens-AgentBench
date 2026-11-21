import numpy as np
import pandas as pd

from crisislens_agentbench.audit import audit, norm_text, to_markdown
from crisislens_agentbench.benchmark import choose_thresholds, grouped_bootstrap, grouped_resplit, run_task
from crisislens_agentbench.humset import _as_list, find_split_files, normalise


def toy(n=60):
    rng = np.random.default_rng(0)
    words = {"Health": "clinic cholera hospital", "WASH": "water latrine hygiene", "Shelter": "tent camp roof"}
    rows = []
    for i in range(n):
        labs = list(rng.choice(list(words), size=rng.integers(1, 3), replace=False))
        text = " ".join(words[l] for l in labs) + f" report {i}"
        split = "train" if i < 36 else ("validation" if i < 48 else "test")
        rows.append({"entry_id": str(i), "lead_id": f"d{i // 3}", "project_id": "p1", "lang": ["en", "fr", "es"][i % 3],
                     "excerpt": text, "split": split, "sectors": labs, "pillars_1d": [], "pillars_2d": labs[:1],
                     "subpillars_1d": [], "subpillars_2d": []})
    return pd.DataFrame(rows)


def test_label_parsing():
    assert _as_list("['Health', 'WASH']") == ["Health", "WASH"]
    assert _as_list(None) == [] and _as_list("[]") == [] and _as_list(["a"]) == ["a"]


def test_find_split_files_prefers_newest_version(tmp_path):
    for v in ("1.0.0", "2.0.0"):
        d = tmp_path / "data" / v
        d.mkdir(parents=True)
        for s in ("train", "validation", "test"):
            (d / f"{s}.jsonl").write_text("{}\n")
    files = find_split_files(tmp_path)
    assert all("2.0.0" in str(p) for p in files.values())


def test_normalise_language_code():
    frame = pd.DataFrame({"entry_id": [1], "lead_id": [2], "project_id": [3], "lang": ["English"], "excerpt": ["x"],
                          "sectors": ["['Health']"]})
    out = normalise(frame, "train")
    assert out.loc[0, "lang"] == "en" and out.loc[0, "sectors"] == ["Health"] and out.loc[0, "pillars_1d"] == []


def test_audit_finds_document_leakage():
    df = toy()
    df.loc[df["split"] == "test", "lead_id"] = "d0"
    rep = audit(df)
    assert rep["cross_split_overlap"]["train->test"]["documents"]["share_of_b_rows"] == 1.0
    assert "Cross-split overlap" in to_markdown(rep)
    assert norm_text("Água,  Potável!") == "agua potavel"


def test_thresholds_and_bootstrap():
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    p = np.array([[0.9, 0.2], [0.3, 0.7], [0.8, 0.1], [0.2, 0.6]])
    th = choose_thresholds(y, p)
    assert ((p >= th) == y).all()
    ci = grouped_bootstrap(y, (p >= th).astype(int), np.array(["a", "a", "b", "b"]), n_boot=50)
    assert ci["micro_f1"] == [1.0, 1.0]


def test_grouped_resplit_has_no_shared_documents():
    out = grouped_resplit(toy(90))
    sets = [set(out.loc[out["split"] == s, "lead_id"]) for s in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_tfidf_baseline_learns_toy_labels():
    def small(seed=0):
        from crisislens_agentbench.benchmark import tfidf_model
        return tfidf_model(seed=seed, max_features=1000, n_jobs=1)
    r = run_task(toy(), "sectors", n_boot=20, model_factory=small)
    assert r["tfidf_lr"]["test"]["micro_f1"] > r["prior"]["test"]["micro_f1"]
    assert set(r["tfidf_lr"]["by_language"]) == {"en", "fr", "es"}


def test_normalise_falls_back_to_document_url():
    frame = pd.DataFrame({"entry_id": [1, 2], "document": ["http://a", "http://a"], "lang": ["es", "es"],
                          "excerpt": ["x", "y"], "sectors": [["Health"], []]})
    out = normalise(frame, "test")
    assert list(out["lead_id"]) == ["http://a", "http://a"] and out.attrs["document_column"] == "document"


def test_scores_undefined_cases_return_nan():
    import math
    from crisislens_agentbench.benchmark import scores
    assert math.isnan(scores(np.zeros((0, 3), int), np.zeros((0, 3), int))["micro_f1"])
    assert math.isnan(scores(np.zeros((4, 2), int), np.zeros((4, 2), int))["macro_f1"])


def test_find_split_files_uses_official_names_and_ignores_bias_files(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for name in ("train_1_1", "validation_1_1", "test_1_1", "train", "validation", "test", "test_gender",
                 "test_country", "humset_bias_train", "humset_bias_val"):
        (d / f"{name}.jsonl").write_text("{}\n")
    files = find_split_files(tmp_path)
    assert {s: p.name for s, p in files.items()} == {"train": "train_1_1.jsonl", "validation": "validation_1_1.jsonl",
                                                     "test": "test_1_1.jsonl"}
    (d / "test_1_1.jsonl").unlink()
    files = find_split_files(tmp_path)
    assert "gender" not in files["test"].name and "country" not in files["test"].name
