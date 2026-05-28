"""Turn cached DBLP + Semantic Scholar JSON into one tidy parquet per venue.

Output: data/interim/<venue>_papers.parquet with columns
    paper_id, title, year, venue, doi, dblp_key, has_abstract,
    abstract, text  (= normalized title + abstract, ready for TF-IDF)
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata

import pandas as pd

from src.utils import (DATA_INTERIM, DATA_RAW, ensure_dirs, load_venues,
                       setup_logging, slug_to_filename)

log = setup_logging("clean")

# Match LaTeX math like $x^2$ and HTML tags so we can strip them.
LATEX = re.compile(r"\$[^$]*\$")
HTML = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    """Lowercase, drop LaTeX/HTML, collapse whitespace. Empty string for None."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)  # canonical unicode form
    s = LATEX.sub(" ", s)
    s = HTML.sub(" ", s)
    return WS.sub(" ", s.lower()).strip()


def load_raw(venue_key: str) -> pd.DataFrame:
    """Read every cached DBLP file for the venue and return one row per paper."""
    cfg = load_venues()[venue_key]
    rows = []
    for slug in cfg["dblp_slugs"]:
        for year in range(cfg["start_year"], cfg["end_year"] + 1):
            cache = DATA_RAW / f"{slug_to_filename(slug)}_{year}.json"
            if not cache.exists():
                continue
            blob = json.loads(cache.read_text())
            for hit in blob.get("result", {}).get("hits", {}).get("hit", []):
                info = hit.get("info") or {}
                doi = (info.get("doi") or "").strip().lower() or None
                rows.append({
                    "paper_id": hit.get("@id") or doi or info.get("key"),
                    "title": info.get("title") or "",
                    "year": int(info["year"]) if info.get("year") else None,
                    "venue": venue_key,
                    "doi": doi,
                    "dblp_key": info.get("key") or "",
                })
    log.info("%s: %d raw rows", venue_key, len(rows))
    return pd.DataFrame(rows)


def attach_abstracts(df: pd.DataFrame, venue_key: str) -> pd.DataFrame:
    """Merge Semantic Scholar abstracts by DOI; missing DOIs simply get None."""
    cache = DATA_RAW / f"s2_{venue_key}.json"
    s2 = json.loads(cache.read_text()) if cache.exists() else {}
    df = df.copy()
    df["abstract"] = df["doi"].map(lambda d: (s2.get(d) or {}).get("abstract") if d else None)
    df["has_abstract"] = df["abstract"].fillna("").str.len() > 0
    pct = 100 * df["has_abstract"].mean() if len(df) else 0
    log.info("%s: %d/%d papers have abstracts (%.1f%%)",
             venue_key, int(df["has_abstract"].sum()), len(df), pct)
    return df


def filter_workshops(df: pd.DataFrame) -> pd.DataFrame:
    """Drop workshop/companion proceedings.

    DBLP keys for the main track look like conf/icse/AuthorYY; workshop papers
    live under conf/icse/icseW or conf/icse/icseC. We drop anything whose
    second-to-last path segment ends in 'W' or 'C' (uppercase only — avoids
    nuking conferences whose names happen to end in 'c').
    """
    sub = df["dblp_key"].str.split("/").str[-2].fillna("")
    keep = ~sub.str.endswith(("W", "C"))
    dropped = (~keep).sum()
    if dropped:
        log.info("dropped %d workshop/companion rows", int(dropped))
    return df[keep].reset_index(drop=True)


def dedupe_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-title duplicates within a year (DBLP occasionally double-lists)."""
    before = len(df)
    key = df["title"].fillna("").str.lower().str.strip()
    df = df.loc[~key.duplicated() | df["year"].duplicated()].copy()
    # Belt-and-braces: also dedupe by DBLP key when present.
    df = df.drop_duplicates(subset=["dblp_key"], keep="first").reset_index(drop=True)
    log.info("dedupe: %d -> %d", before, len(df))
    return df


def clean_venue(venue_key: str) -> None:
    """Run the whole cleaning pipeline and write the interim parquet."""
    ensure_dirs()
    df = load_raw(venue_key)
    df = filter_workshops(df)
    df = attach_abstracts(df, venue_key)
    df = dedupe_titles(df)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    # 'text' is what analyse.py will vectorise — normalised title+abstract.
    df["text"] = (df["title"].map(normalize) + " " + df["abstract"].map(normalize)).str.strip()
    out = DATA_INTERIM / f"{venue_key}_papers.parquet"
    df.to_parquet(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))


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
        clean_venue(v)


if __name__ == "__main__":
    main()
