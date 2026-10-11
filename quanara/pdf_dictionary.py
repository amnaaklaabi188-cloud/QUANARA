"""Verbatim column entries from Data_Dictionary.pdf.

The PDF lays each data domain out as a four-column table (COLUMN NAME · DATA TYPE · DESCRIPTION ·
EXAMPLE VALUES) without row rules, so rows are reconstructed from word positions: a row starts on
the line where both the column-name and the data-type columns begin a new entry. The text of each
cell is returned exactly as printed (line breaks joined with spaces), with its page number, so the
app can show the source definition next to — and separately from — its own modelling decisions.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from . import config

SECTIONS = {"International Guest Data": "international", "Domestic Guest Data": "domestic", "Flight Operations Data": "flight"}


def _lines(words: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if out and abs(out[-1][0]["top"] - w["top"]) < 2.5:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def _join(ws: list[dict]) -> str:
    return " ".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"]))


@lru_cache(maxsize=2)
def _entries_cached(path: str) -> tuple:
    import pdfplumber
    entries, section, cols = [], None, None
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
            for line in _lines(words):
                text = _join(line)
                if text in SECTIONS:
                    section, cols = SECTIONS[text], None  # each domain has its own table header
                    continue
                if text.startswith("COLUMN NAME") and "DESCRIPTION" in text:
                    pos = {w["text"]: w["x0"] for w in line}
                    cols = (pos["DATA"] - 3, pos["DESCRIPTION"] - 3, pos["EXAMPLE"] - 3)
                    continue
                if cols is None or section is None or text.startswith("Data Dictionary v"):
                    continue
                name = [w for w in line if w["x0"] < cols[0]]
                typ = [w for w in line if cols[0] <= w["x0"] < cols[1]]
                desc = [w for w in line if cols[1] <= w["x0"] < cols[2]]
                ex = [w for w in line if w["x0"] >= cols[2]]
                starts = bool(name) and bool(typ) and not _join(typ).startswith("(")
                if starts:
                    entries.append({"section": section, "page": pno, "name": [], "type": [], "description": [], "examples": []})
                if not entries or entries[-1]["section"] != section:
                    continue
                e = entries[-1]
                for k, ws in (("name", name), ("type", typ), ("description", desc), ("examples", ex)):
                    if ws:
                        e[k].append(_join(ws))
    out = []
    for e in entries:
        d = {k: re.sub(r"\s+", " ", " ".join(e[k])).strip() for k in ("name", "type", "description", "examples")}
        # the PDF prints the asterisk as a separate glyph on its own line; restore it where a sentence refers to it
        d["description"] = re.sub(r"(?<=\s)(?=indicates|=)", "", d["description"])
        out.append({"section": e["section"], "page": e["page"], **d})
    return tuple(tuple(sorted(x.items())) for x in out)


def entries(path: Path | None = None) -> list[dict]:
    """All column entries: section, page, name, type, description, examples (verbatim)."""
    p = path or next(config.RAW_DIR.glob("*[Dd]ictionary*.pdf"))
    return [dict(t) for t in _entries_cached(str(p))]


def lookup(column: str, section: str) -> dict | None:
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    for e in entries():
        if e["section"] == section and norm(e["name"]) == norm(column):
            return e
    return None
