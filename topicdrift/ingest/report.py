"""
report.py — Build the human-readable preview CSV.

Reads:  data/interim/<venue>_enriched.parquet  (from src/ingest/enrich_openalex.py)
Writes: outputs/tables/<venue>_papers_preview.csv

Same row-grain as silver, denormalised and sorted so a person can skim it in
a spreadsheet:
  - drop noisy provenance columns (openalex_id, oa_type, dblp_id, dblp_key, ee, url, text)
  - rename oa_concepts -> keywords for clarity
  - semicolon-join list columns so the CSV opens cleanly
  - strip leaked XML/HTML markup from abstracts (OpenAlex embeds <tex> sometimes)
  - sort newest-first, with abstract-present rows on top within each year
"""

import re
import sys
from pathlib import Path

import pandas as pd

INTERIM_DIR = Path("data/interim")
OUTPUTS_TABLES = Path("outputs/tables")
OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)

PREVIEW_COLUMNS = [
    "year",
    "venue",
    "title",
    "authors",
    "abstract",
    "keywords",
    "citation_count",
    "doi",
    "has_abstract",
]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_abstract(text) -> str:
    if not text:
        return ""
    s = _TAG_RE.sub(" ", str(text))
    return _WS_RE.sub(" ", s).strip()


def _join_list(value) -> str:
    if value is None:
        return ""
    try:
        items = list(value)
    except TypeError:
        return str(value)
    return "; ".join(str(x) for x in items) if items else ""


def build_report(venue_key: str) -> None:
    src = INTERIM_DIR / f"{venue_key}_enriched.parquet"
    if not src.exists():
        raise SystemExit(f"Not found: {src}. Run `make venue VENUE={venue_key}` first.")

    df = pd.read_parquet(src)
    print(f"Loaded {len(df)} rows from {src}")

    out = df.rename(columns={"oa_concepts": "keywords"}).copy()
    out = out[[c for c in PREVIEW_COLUMNS if c in out.columns]]
    out = out.sort_values(
        ["year", "has_abstract", "title"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    out["authors"] = out["authors"].apply(_join_list)
    if "keywords" in out.columns:
        out["keywords"] = out["keywords"].apply(_join_list)
    out["abstract"] = out["abstract"].apply(_clean_abstract)

    dest = OUTPUTS_TABLES / f"{venue_key}_papers_preview.csv"
    out.to_csv(dest, index=False)
    pct = 100 * out["has_abstract"].mean() if "has_abstract" in out.columns else 0
    print(f"Wrote {dest} ({len(out)} rows, {pct:.1f}% with abstracts)")


if __name__ == "__main__":
    venues = sys.argv[1:] or ["icse"]
    for v in venues:
        build_report(v)
