"""
clean.py — Turn cached DBLP JSON into a tidy parquet file /interim.

Steps:
1. Parse raw DBLP JSON batches
2. Flatten each hit into a row
3. Filter to main track papers only
4. Deduplicate
5. Flag missing DOIs
6. Output data/interim/icse_dblp.parquet
"""

import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_dblp(venue_key: str) -> list:
    """Load all cached DBLP per-year files for a venue."""
    hits = []
    for path in sorted(RAW_DIR.glob(f"{venue_key}_*.json")):
        hits.extend(json.loads(path.read_text()))
    print(f"Loaded {len(hits)} raw hits for {venue_key}")
    return hits


def parse_hits(hits: list) -> pd.DataFrame:
    """Flatten each DBLP hit into a row."""
    rows = []
    for hit in hits:
        info = hit.get("info", {})

        # authors can be a list or a single dict
        authors_raw = info.get("authors", {}).get("author", [])
        if isinstance(authors_raw, dict):
            authors_raw = [authors_raw]
        authors = [a.get("text", "") for a in authors_raw]

        rows.append({
            "dblp_id":  hit.get("@id"),
            "dblp_key": info.get("key"),
            "title":    info.get("title", "").rstrip("."),
            "year":     int(info["year"]) if info.get("year") else None,
            "doi":      info.get("doi", "").strip().lower() or None,
            "authors":  authors,
            "url":      info.get("url"),
            "ee":        info.get("ee"),
        })
    return pd.DataFrame(rows)


def filter_main_track(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only main track papers. Workshop/companion keys end in W or C."""
    sub = df["dblp_key"].str.split("/").str[-2].fillna("")
    keep = ~sub.str.endswith(("W", "C"))
    dropped = (~keep).sum()
    print(f"Filtered {dropped} workshop/companion papers")
    return df[keep].reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by dblp_key, then by title+year. Warn on any duplicates found."""
    before = len(df)

    key_dupes = df[df.duplicated(subset=["dblp_key"], keep=False)]
    if not key_dupes.empty:
        print(f"  WARNING: {len(key_dupes)} rows share a duplicate dblp_key:")
        for key, group in key_dupes.groupby("dblp_key"):
            print(f"    {key}: {list(group['title'])}")

    df = df.drop_duplicates(subset=["dblp_key"], keep="first")

    title_dupes = df[df.duplicated(subset=["title", "year"], keep=False)]
    if not title_dupes.empty:
        print(f"  WARNING: {len(title_dupes)} rows share a duplicate title+year:")
        for (title, year), group in title_dupes.groupby(["title", "year"]):
            print(f"    [{year}] {title}: dblp_keys={list(group['dblp_key'])}")
        dupes_path = INTERIM_DIR / "icse_duplicates.csv"
        title_dupes.sort_values(["year", "title"]).to_csv(dupes_path, index=False)
        print(f"  Wrote {len(title_dupes)} duplicate rows to {dupes_path}")

    # Prefer rows with DOI, then with more authors, before dropping duplicates.
    df = df.assign(
        _has_doi=df["doi"].notna(),
        _n_authors=df["authors"].map(len),
    ).sort_values(["_has_doi", "_n_authors"], ascending=[False, False])
    df = df.drop_duplicates(subset=["title", "year"], keep="first")
    df = df.drop(columns=["_has_doi", "_n_authors"]).reset_index(drop=True)

    print(f"Deduplicated: {before} -> {len(df)} papers")
    return df


def flag_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Add has_doi flag. Papers without DOI can't be matched to Semantic Scholar."""
    df = df.copy()
    df["has_doi"] = df["doi"].notna()
    missing = (~df["has_doi"]).sum()
    print(f"Papers missing DOI: {missing} ({100 * missing / len(df):.1f}%)")
    return df


def clean_venue(venue_key: str) -> None:
    hits = load_raw_dblp(venue_key)
    df = parse_hits(hits)
    df = filter_main_track(df)
    df = deduplicate(df)
    df = flag_missing(df)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    out = INTERIM_DIR / f"{venue_key}_dblp.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    clean_venue("icse")