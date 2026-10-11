"""Locate and read the competition workbooks. Files are opened read-only and never modified.

Parsed workbooks are cached as pickles in data/cache keyed by the source file's SHA-256, so
the cache can never serve stale data after a file changes.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from . import config
from .cleaning import clean_dataset
from .schema import validate


class DataFileNotFound(FileNotFoundError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_file(pattern: str, raw_dir: Path | None = None) -> Path:
    raw_dir = Path(raw_dir or config.RAW_DIR)
    if not raw_dir.exists():
        raise DataFileNotFound(f"Data folder not found: {raw_dir}")
    rx = re.compile(pattern, re.IGNORECASE)
    hits = sorted(p for p in raw_dir.iterdir() if p.is_file() and rx.search(p.name) and not p.name.startswith("~$"))
    if not hits:
        raise DataFileNotFound(f"No file matching /{pattern}/ in {raw_dir}")
    if len(hits) > 1:
        # Prefer the shortest name (the un-suffixed original) but never silently pick between
        # two different files: if contents differ, fail loudly.
        digests = {sha256(p) for p in hits}
        if len(digests) > 1:
            raise ValueError(f"Ambiguous: several different files match /{pattern}/: {[h.name for h in hits]}")
        hits.sort(key=lambda p: len(p.name))
    return hits[0]


def discover_files(raw_dir: Path | None = None) -> dict[str, Path]:
    files = {k: find_file(p, raw_dir) for k, p in config.DATASET_PATTERNS.items()}
    files["dictionary"] = find_file(config.DICTIONARY_PATTERN, raw_dir)
    return files


def read_raw(dataset: str, raw_dir: Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Read one workbook exactly as stored (object dtype; empty cells -> NaN)."""
    path = find_file(config.DATASET_PATTERNS[dataset], raw_dir)
    digest = sha256(path)[:16]
    cache = config.CACHE_DIR / f"{dataset}_{digest}.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)
    df = pd.read_excel(path, sheet_name=config.SHEET_NAME, dtype=object)
    if use_cache:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


def load_dataset(dataset: str, raw_dir: Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Read, validate and clean one dataset."""
    raw = read_raw(dataset, raw_dir, use_cache)
    df = clean_dataset(dataset, raw)
    res = validate(dataset, df)
    if not res.ok:
        raise ValueError(f"Schema validation failed for {dataset}: {res.messages}")
    return df


def load_all(raw_dir: Path | None = None, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    return {k: load_dataset(k, raw_dir, use_cache) for k in config.DATASET_PATTERNS}


def file_inventory(raw_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    for key, path in discover_files(raw_dir).items():
        rows.append({"dataset": key, "file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return pd.DataFrame(rows)
