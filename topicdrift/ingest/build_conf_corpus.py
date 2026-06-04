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
import logging
from pathlib import Path

import pandas as pd

from topicdrift.ingest._matching import normalize
from topicdrift.ingest.enrich_openalex import reconstruct_abstract
from topicdrift.utils.doi import normalize_doi

log = logging.getLogger(__name__)

INTERIM_DIR = Path("data/interim")
OA_SCAN_CACHE = Path("data/raw/openalex_scan")

SRC = INTERIM_DIR / "dblp_conf.parquet"
DEST = INTERIM_DIR / "conf_enriched.parquet"


def _venue_abstract_map() -> dict[str, str]:
    """dblp_key -> abstract from the per-venue Workflow-A silver parquets.

    The DOI scan only recovers abstracts for papers DBLP has a DOI for, which
    excludes most pre-1990 conference papers. The per-venue `make venue-deep`
    pipeline (enrich_openalex --title-pass + ACM scrape) recovers abstracts for
    those DOI-less rows by title match. Those parquets are keyed by the same
    DBLP key, so we can fold their abstracts back into the pooled corpus to fill
    gaps the scan missed. Returns {} when no venue parquets are present."""
    out: dict[str, str] = {}
    for path in sorted(INTERIM_DIR.glob("*_enriched.parquet")):
        if path.name == DEST.name:
            continue
        cols = pd.read_parquet(path, columns=["dblp_key", "abstract"])
        cols = cols[cols["abstract"].fillna("").str.len() > 0]
        out.update(dict(zip(cols["dblp_key"], cols["abstract"])))
    if out:
        log.info("  venue-abstract fallback: %d keys from per-venue parquets", len(out))
    return out


def _build_doi_abstract_map() -> dict[str, str]:
    """Read every scan batch file; return doi (normalized) -> reconstructed abstract."""
    batch_files = [p for p in OA_SCAN_CACHE.glob("*.json") if p.name != "_scan_summary.json"]
    total = len(batch_files)
    log.info("Reading %d scan batch files...", total)
    out: dict[str, str] = {}
    for i, path in enumerate(batch_files, 1):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for work in data.get("results", []):
            doi = normalize_doi(work.get("doi"))
            if not doi:
                continue
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if abstract:
                out[doi] = abstract
        if i % 5_000 == 0:
            log.info("  %d/%d files, %d abstracts", i, total, len(out))
    log.info("  done: %d DOIs with abstract text", len(out))
    return out


def build() -> None:
    if not SRC.exists():
        raise SystemExit(f"Not found: {SRC}. Run `make dump` first.")

    df = pd.read_parquet(SRC)
    log.info("Loaded %d papers from %s", len(df), SRC)

    doi_abstract = _build_doi_abstract_map()

    doi_norm = df["doi"].map(normalize_doi)
    df["abstract"] = doi_norm.map(doi_abstract)
    df["has_abstract"] = df["abstract"].fillna("").str.len() > 0

    # Fallback: fill abstracts the DOI scan missed from the per-venue silver
    # parquets (recovers DOI-less early-conference papers, e.g. ICSE pre-1990).
    venue_abs = _venue_abstract_map()
    if venue_abs:
        missing = ~df["has_abstract"]
        df.loc[missing, "abstract"] = df.loc[missing, "dblp_key"].map(venue_abs)
        df["has_abstract"] = df["abstract"].fillna("").str.len() > 0
        n_filled = int((missing & df["has_abstract"]).sum())
        log.info("  filled %d previously-missing abstracts from venue parquets", n_filled)

    df["text"] = (df["title"].map(normalize) + " " + df["abstract"].map(normalize)).str.strip()

    out = df[["conf", "dblp_key", "title", "year", "doi", "abstract", "has_abstract", "text"]]
    out.to_parquet(DEST, index=False)
    n_with = int(df["has_abstract"].sum())
    log.info(
        "Wrote %s (%d rows, %d with abstract, %.1f%% coverage)",
        DEST,
        len(out),
        n_with,
        100 * n_with / len(out),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
