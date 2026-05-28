"""Enrich the interim papers parquet with OpenAlex abstracts + concepts.

Reads:  data/interim/<venue>_papers.parquet  (from src.clean)
Writes: data/interim/<venue>_papers_enriched.parquet

Adds columns:
    abstract        str | None    reconstructed from OpenAlex inverted index
    has_abstract    bool          recomputed from `abstract`
    text            str           normalized title + abstract (TF-IDF ready)
    oa_concepts     list[str]     concept tags with score >= 0.3
    citation_count  int | None    cited_by_count
    openalex_id     str | None
    oa_type         str | None    e.g. "article"

OpenAlex is free; passing `mailto` puts us in the polite pool. Batches DOIs
50 at a time via the OR-filter. Raw responses cached under data/raw/openalex/
so re-runs cost nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

from src.utils import DATA_INTERIM, DATA_RAW, ensure_dirs, load_venues, setup_logging

log = setup_logging("enrich_openalex")

OA_CACHE = DATA_RAW / "openalex"
OPENALEX_URL = "https://api.openalex.org/works"
MAILTO = "maaseide.m@northeastern.edu"
BATCH_SIZE = 50
SLEEP_BETWEEN_CALLS = 0.15
CONCEPT_MIN_SCORE = 0.3
SELECT_FIELDS = "id,doi,abstract_inverted_index,concepts,cited_by_count,type"

LATEX = re.compile(r"\$[^$]*\$")
HTML = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    """Mirror src.clean.normalize so `text` column is consistent."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = LATEX.sub(" ", s)
    s = HTML.sub(" ", s)
    return WS.sub(" ", s.lower()).strip()


def reconstruct_abstract(inv_index: dict | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions...]} — rebuild the text."""
    if not inv_index:
        return None
    positions = [(i, w) for w, idxs in inv_index.items() for i in idxs]
    positions.sort()
    return " ".join(w for _, w in positions) or None


def _batch_cache_path(dois: list[str]) -> Path:
    digest = hashlib.sha1("|".join(sorted(dois)).encode()).hexdigest()
    return OA_CACHE / f"{digest}.json"


def _fetch_batch(dois: list[str]) -> list[dict]:
    OA_CACHE.mkdir(parents=True, exist_ok=True)
    cache = _batch_cache_path(dois)
    if cache.exists():
        return json.loads(cache.read_text()).get("results", [])

    params = {
        "filter": "doi:" + "|".join(dois),
        "per-page": BATCH_SIZE,
        "mailto": MAILTO,
        "select": SELECT_FIELDS,
    }
    for attempt in range(4):
        r = requests.get(OPENALEX_URL, params=params, timeout=30)
        if r.status_code == 429:
            wait = 2 ** attempt * 3
            log.warning("OpenAlex rate-limited; sleeping %ds", wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        break
    else:
        log.error("OpenAlex gave up after retries (batch of %d)", len(dois))
        return []

    payload = r.json()
    cache.write_text(json.dumps(payload, ensure_ascii=False))
    time.sleep(SLEEP_BETWEEN_CALLS)
    return payload.get("results", [])


def parse_work(work: dict) -> dict:
    raw_doi = (work.get("doi") or "").replace("https://doi.org/", "").lower() or None
    concepts = [
        c["display_name"]
        for c in (work.get("concepts") or [])
        if (c.get("score") or 0) >= CONCEPT_MIN_SCORE
    ]
    return {
        "doi": raw_doi,
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "oa_concepts": concepts,
        "citation_count": work.get("cited_by_count"),
        "openalex_id": work.get("id"),
        "oa_type": work.get("type"),
    }


def fetch_for_dois(dois: list[str]) -> dict[str, dict]:
    uniq = sorted({d for d in dois if d})
    n_batches = (len(uniq) + BATCH_SIZE - 1) // BATCH_SIZE
    log.info("%d unique DOIs in %d batches of %d", len(uniq), n_batches, BATCH_SIZE)

    out: dict[str, dict] = {}
    for i in range(0, len(uniq), BATCH_SIZE):
        batch = uniq[i : i + BATCH_SIZE]
        if (i // BATCH_SIZE) % 20 == 0:
            log.info("batch %d/%d", i // BATCH_SIZE + 1, n_batches)
        for work in _fetch_batch(batch):
            parsed = parse_work(work)
            if parsed["doi"]:
                out[parsed["doi"]] = parsed
    log.info("OpenAlex returned metadata for %d / %d DOIs", len(out), len(uniq))
    return out


def enrich_venue(venue_key: str) -> None:
    ensure_dirs()
    src = DATA_INTERIM / f"{venue_key}_papers.parquet"
    if not src.exists():
        raise SystemExit(f"Not found: {src}. Run `python -m src.clean --venue {venue_key}` first.")

    df = pd.read_parquet(src)
    log.info("read %s (%d rows)", src, len(df))

    enriched = fetch_for_dois(df["doi"].dropna().tolist())
    empty = {"abstract": None, "oa_concepts": [], "citation_count": None,
             "openalex_id": None, "oa_type": None}

    rows = [
        {k: v for k, v in (enriched.get(d) or empty).items() if k != "doi"}
        if d else empty.copy()
        for d in df["doi"]
    ]
    add = pd.DataFrame(rows)

    # Drop any columns that already exist (abstract/has_abstract were stubbed in clean.py).
    for col in add.columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    out = pd.concat([df.reset_index(drop=True), add.reset_index(drop=True)], axis=1)

    # Recompute downstream-derived columns.
    out["has_abstract"] = out["abstract"].fillna("").str.len() > 0
    out["text"] = (out["title"].map(normalize) + " " + out["abstract"].map(normalize)).str.strip()

    dest = DATA_INTERIM / f"{venue_key}_papers_enriched.parquet"
    out.to_parquet(dest, index=False)
    log.info("wrote %s (%d rows)", dest, len(out))

    pct = 100 * out["has_abstract"].mean()
    log.info("abstract coverage: %d / %d (%.1f%%)",
             int(out["has_abstract"].sum()), len(out), pct)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--venue")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    venues = list(load_venues())
    targets = venues if args.all else [args.venue] if args.venue else []
    if not targets:
        p.error("pass --venue <key> or --all")
    for v in targets:
        enrich_venue(v)


if __name__ == "__main__":
    main()
