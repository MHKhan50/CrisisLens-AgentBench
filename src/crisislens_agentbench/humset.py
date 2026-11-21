"""HumSet loader: download at run time from the Hugging Face Hub, record provenance, normalise labels.

HumSet (Fekih et al., Findings of EMNLP 2022) holds expert-annotated excerpts from humanitarian response documents
in English, French and Spanish, labelled with the DEEP analytical framework. The data are not redistributed here.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ID = "nlp-thedeep/humset"
TASKS = ["sectors", "pillars_1d", "pillars_2d", "subpillars_1d", "subpillars_2d"]
SPLITS = ["train", "validation", "test"]
_SPLIT_ALIASES = {"train": ("train",), "validation": ("validation", "val", "dev"), "test": ("test",)}
# Official file names per dataset version (the dataset's own loading script, humset.py on the Hub).
OFFICIAL_FILES = {
    "2.0.0": {"train": "data/train_1_1.jsonl", "validation": "data/validation_1_1.jsonl", "test": "data/test_1_1.jsonl"},
    "1.0.0": {"train": "data/train.jsonl", "validation": "data/validation.jsonl", "test": "data/test.jsonl"},
}
DEFAULT_VERSION = "2.0.0"
# Other files in the same folder that are not HumSet splits (HumSet-Bias material).
_NOT_SPLITS = ("bias", "gender", "country")
_READERS = {".csv": "csv", ".tsv": "csv", ".jsonl": "json", ".json": "json", ".parquet": "parquet"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _suffix(path: Path) -> str:
    suffixes = [s for s in path.suffixes if s not in (".gz", ".zip", ".bz2", ".xz")]
    return suffixes[-1] if suffixes else ""


def _read(path: Path) -> pd.DataFrame:
    kind = _READERS[_suffix(path)]
    if kind == "csv":
        return pd.read_csv(path, sep="\t" if _suffix(path) == ".tsv" else ",")
    if kind == "parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)


def _version_key(path: Path) -> tuple:
    """Sort key that prefers the highest version folder (e.g. data/2.0.0 over data/1.0.0)."""
    for part in reversed(path.parts):
        bits = part.split(".")
        if len(bits) >= 2 and all(b.isdigit() for b in bits):
            return tuple(int(b) for b in bits)
    return (0,)


def find_split_files(root: Path, version: str = DEFAULT_VERSION) -> Dict[str, Path]:
    """Map train/validation/test to data files under ``root``.

    The official file names of ``version`` are used when present. Otherwise files are matched by name (newest version
    folder first), ignoring the HumSet-Bias files (``*bias*``, ``test_gender``, ``test_country``)."""
    official = {s: root / p for s, p in OFFICIAL_FILES.get(version, {}).items()}
    if official and all(p.is_file() for p in official.values()):
        return official
    candidates = [p for p in root.rglob("*") if p.is_file() and _suffix(p) in _READERS
                  and not any(w in p.name.lower() for w in _NOT_SPLITS)]
    found: Dict[str, Path] = {}
    for split, aliases in _SPLIT_ALIASES.items():
        matches = [p for p in candidates if any(a in p.name.lower().split(".")[0].replace("-", "_").split("_")
                                                or p.name.lower().startswith(a) for a in aliases)]
        if matches:
            found[split] = sorted(matches, key=lambda p: (_version_key(p), str(p)))[-1]
    missing = [s for s in SPLITS if s not in found]
    if missing:
        raise FileNotFoundError(f"HumSet split files not found for {missing} under {root}; files: "
                                f"{sorted(str(p.relative_to(root)) for p in candidates)}")
    return found


def download(data_dir: str = "data/humset", revision: str = "main") -> Path:
    """Download the HumSet data files from the Hugging Face Hub into ``data_dir`` and write a manifest."""
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().dataset_info(REPO_ID, revision=revision)
    root = Path(snapshot_download(REPO_ID, repo_type="dataset", revision=info.sha, local_dir=data_dir,
                                  allow_patterns=[*OFFICIAL_FILES[DEFAULT_VERSION].values(), "README.md"]))
    files = find_split_files(root)
    manifest = {
        "dataset": REPO_ID, "version": DEFAULT_VERSION, "revision": info.sha, "licence": "Apache-2.0 (dataset card)",
        "citation": "Fekih et al., HumSet, Findings of EMNLP 2022",
        "files": {s: {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": _sha256(p)}
                  for s, p in files.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def _as_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if hasattr(value, "tolist"):
        return [str(v) for v in value.tolist()]
    text = str(value).strip()
    if not text or text in ("[]", "nan"):
        return []
    if text.startswith("["):
        try:
            return [str(v) for v in ast.literal_eval(text)]
        except (ValueError, SyntaxError):
            pass
    return [t.strip() for t in text.split(";") if t.strip()]


_ALIASES = {
    "entry_id": ("entry_id", "id"),
    "lead_id": ("lead_id", "lead", "document_id", "doc_id", "document"),
    "project_id": ("project_id", "project"),
    "lang": ("lang", "language"),
    "excerpt": ("excerpt", "text", "modified_excerpt"),
}


def _column(frame: pd.DataFrame, field: str):
    for name in _ALIASES[field]:
        if name in frame.columns:
            return name
    return None


def normalise(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Keep the columns used here, with label columns as Python lists of strings.

    The document id falls back to the source-document URL when ``lead_id`` is absent, and to the entry id as a
    last resort (then document-level grouping is not possible and the audit says so).
    """
    cols = {f: _column(frame, f) for f in _ALIASES}
    if cols["excerpt"] is None:
        raise KeyError(f"no excerpt column in HumSet file; columns: {list(frame.columns)}")
    entry = frame[cols["entry_id"]].astype(str) if cols["entry_id"] else pd.Series(
        [f"{split}-{i}" for i in range(len(frame))], index=frame.index)
    out = pd.DataFrame({
        "entry_id": entry,
        "lead_id": frame[cols["lead_id"]].astype(str) if cols["lead_id"] else entry,
        "project_id": frame[cols["project_id"]].astype(str) if cols["project_id"] else "",
        "lang": frame[cols["lang"]].astype(str).str.lower().str[:2] if cols["lang"] else "xx",
        "excerpt": frame[cols["excerpt"]].astype(str),
        "split": split,
    })
    out.attrs["document_column"] = cols["lead_id"] or "none (entry id used)"
    for task in TASKS:
        out[task] = frame[task].map(_as_list) if task in frame else [[] for _ in range(len(frame))]
    return out


def load(root: str, limit: Optional[int] = None) -> pd.DataFrame:
    """All three official splits as one frame with a ``split`` column."""
    import sys
    files = find_split_files(Path(root))
    parts = []
    for split in SPLITS:
        frame = _read(files[split])
        print(f"{split}: {files[split].relative_to(root)} rows={len(frame)} columns={list(frame.columns)}",
              file=sys.stderr, flush=True)
        if limit:
            frame = frame.head(limit)
        parts.append(normalise(frame, split))
    out = pd.concat(parts, ignore_index=True)
    out.attrs["document_column"] = parts[0].attrs["document_column"]
    out.attrs["data_signature"] = {s: [files[s].name, files[s].stat().st_size, limit] for s in SPLITS}
    print(f"document id column: {out.attrs['document_column']}", file=sys.stderr, flush=True)
    return out
