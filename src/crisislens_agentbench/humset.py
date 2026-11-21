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


def find_split_files(root: Path) -> Dict[str, Path]:
    """Map train/validation/test to data files under ``root``, preferring the newest version folder."""
    candidates = [p for p in root.rglob("*") if p.is_file() and _suffix(p) in _READERS]
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
                                  allow_patterns=["data/*", "data/**", "README.md"]))
    files = find_split_files(root)
    manifest = {
        "dataset": REPO_ID, "revision": info.sha, "licence": "Apache-2.0 (dataset card)",
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


def normalise(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Keep the columns used here, with label columns as Python lists of strings."""
    out = pd.DataFrame({
        "entry_id": frame["entry_id"].astype(str),
        "lead_id": frame["lead_id"].astype(str),
        "project_id": frame["project_id"].astype(str) if "project_id" in frame else "",
        "lang": frame["lang"].astype(str).str.lower().str[:2],
        "excerpt": frame["excerpt"].astype(str),
        "split": split,
    })
    for task in TASKS:
        out[task] = frame[task].map(_as_list) if task in frame else [[] for _ in range(len(frame))]
    return out


def load(root: str, limit: Optional[int] = None) -> pd.DataFrame:
    """All three official splits as one frame with a ``split`` column."""
    files = find_split_files(Path(root))
    parts = []
    for split in SPLITS:
        frame = _read(files[split])
        if limit:
            frame = frame.head(limit)
        parts.append(normalise(frame, split))
    return pd.concat(parts, ignore_index=True)
