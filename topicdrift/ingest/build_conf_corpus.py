"""
build_conf_corpus.py — Assemble the pooled multi-conference corpus offline.

Reads:  data/interim/dblp_conf.parquet          (from ingest_dblp_dump.py)
        data/raw/openalex_scan/*.json             (from conf_abstract_report.py scan)
Writes: data/interim/conf_enriched.parquet

Joins every DBLP conference paper against the OpenAlex scan cache to recover
abstracts. No network calls — all data is already local.

Columns in output:
  conf, dblp_key, title, year, doi, abstract, has_abstract, text
"""
import json
from pathlib import Path

import pandas as pd

from topicdrift.ingest.enrich_openalex import normalize, reconstruct_abstract

INTERIM_DIR = Path("data/interim")
OA_SCAN_CACHE = Path("data/raw/openalex_scan")

SRC = INTERIM_DIR / "dblp_conf.parquet"
DEST = INTERIM_DIR / "conf_enriched.parquet"


def _build_doi_abstract_map() -> dict[str, str]:
    """Read every scan batch file; return doi (normalized) -> reconstructed abstract."""
    batch_files = [p for p in OA_SCAN_CACHE.glob("*.json")
                   if p.name != "_scan_summary.json"]
    total = len(batch_files)
    print(f"Reading {total:,} scan batch files...")
    out: dict[str, str] = {}
    for i, path in enumerate(batch_files, 1):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for work in data.get("results", []):
            doi = (work.get("doi") or "").replace("https://doi.org/", "").lower()
            if not doi:
                continue
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if abstract:
                out[doi] = abstract
        if i % 5_000 == 0:
            print(f"  {i:,}/{total:,} files, {len(out):,} abstracts")
    print(f"  done: {len(out):,} DOIs with abstract text")
    return out


def build() -> None:
    if not SRC.exists():
        raise SystemExit(f"Not found: {SRC}. Run `make dump` first.")

    df = pd.read_parquet(SRC)
    print(f"Loaded {len(df):,} papers from {SRC}")

    doi_abstract = _build_doi_abstract_map()

    # Normalize DOIs the same way the scan cache does.
    doi_norm = (
        df["doi"]
        .fillna("")
        .str.replace("https://doi.org/", "", regex=False)
        .str.lower()
        .replace("", None)
    )
    df["abstract"] = doi_norm.map(doi_abstract)
    df["has_abstract"] = df["abstract"].fillna("").str.len() > 0
    df["text"] = (
        df["title"].map(normalize) + " " + df["abstract"].map(normalize)
    ).str.strip()

    out = df[["conf", "dblp_key", "title", "year", "doi",
              "abstract", "has_abstract", "text"]]
    out.to_parquet(DEST, index=False)
    n_with = int(df["has_abstract"].sum())
    print(f"Wrote {DEST} ({len(out):,} rows, {n_with:,} with abstract, "
          f"{100 * n_with / len(out):.1f}% coverage)")


if __name__ == "__main__":
    build()
