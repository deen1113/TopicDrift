"""
clean.py — Turn cached DBLP JSON into a tidy parquet file /interim.

Steps:
1. Parse raw DBLP JSON batches
2. Flatten each hit into a row
3. Keep only papers via the DBLP type field (drops Editorship proceedings volumes)
4. Filter to main track papers only
5. Deduplicate
6. Flag missing DOIs
7. Output data/interim/icse_dblp.parquet
"""

import html
import json
import re
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
# No conferences in the data
MAIN_TRACK_VENUES = {"ICSE"}

# IEEE assigns 5-digit DOIs ending in 10xxx to proceedings front-matter
# (workshop intros, panel summaries, BoFs) — these have no abstract anywhere.
FRONT_MATTER_DOI = re.compile(r"icse\.\d{4}\.10\d{3}$", re.I)

# Title-shape signals for proceedings front-matter (panels, tutorials,
# workshop overviews, symposia reports). Mix of prefix-anchored and
# anywhere-in-title patterns. Audited against the full corpus: every match
# at the time of writing is a non-research-paper entry — see the run log
# emitted by filter_front_matter() for the dropped list.
FRONT_MATTER_TITLE = re.compile(
    r"(?:"
    # ---- prefix-anchored proceedings shapes ----
    r"^(?:the\s+)?\d+(?:st|nd|rd|th)\s+(?:international\s+|icse\s+)?workshop\b"
    r"|^(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
    r"\s+(?:international\s+|icse\s+)?workshop\b"
    r"|^(?:international\s+)?workshop\s+(?:on|to)\b"
    r"|^icse\s+workshop\b"
    r"|^ecse\s+workshop\b"
    r"|^[a-z]+[\s-]?\d{1,4}\s*[:\-].*\bworkshop\b"  # SCM-10:, FM91:, FM 89:
    r"|^bof:"
    r"|^panel:"
    r"|^tutorial:"
    r"|^keynote:"
    r"|^the\s+international\s+symposium\b"
    r"|^message from\b"  # "Message from the Program Chairs", "Message from the Editors"
    # ---- anywhere-in-title proceedings markers ----
    r"|\bpanel\s+summary\b"
    r"|\(panel[^)]*\)\s*$"    # (panel), (Panel Abstract), (Panel Discussion), etc.
    r"|\(tutorial[^)]*\)\s*$"
    r"|\(workshop\s+report\)"
    r"|\(workshop\s+session\)"
    r"|\bfaculty\s+symposium\b"
    r"|\bdoctoral\s+symposium\b"
    r"|\bparnas\s+symposium\b"
    r"|\bphd\s+symposium\b"
    r")",
    re.I,
)


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

        venue = info.get("venue")
        if isinstance(venue, list):
            venue = venue[0] if venue else None

        rows.append({
            "dblp_id":  hit.get("@id"),
            "dblp_key": info.get("key"),
            "type":     info.get("type"),
            "title":    html.unescape(info.get("title", "")).rstrip("."),
            "year":     int(info["year"]) if info.get("year") else None,
            "doi":      info.get("doi", "").strip().lower() or None,
            "authors":  authors,
            "url":      info.get("url"),
            "ee":       info.get("ee"),
            "venue":    venue,
        })
    return pd.DataFrame(rows)


def filter_main_track(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ICSE-proper (main research track). See MAIN_TRACK_VENUES."""
    keep = df["venue"].isin(MAIN_TRACK_VENUES)
    dropped = (~keep).sum()
    print(f"Filtered {dropped} non-main-track papers, kept {keep.sum()}")
    return df[keep].reset_index(drop=True)


def filter_to_papers(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only DBLP 'Conference and Workshop Papers'; drop 'Editorship'
    proceedings volumes. DBLP's type field is the authoritative signal."""
    keep = df["type"] == "Conference and Workshop Papers"
    dropped = (~keep).sum()
    print(f"Dropped {dropped} non-paper records (by DBLP type)")
    for title in df.loc[~keep, "title"].head(10):
        print(f"    {title[:80]}")
    return df[keep].reset_index(drop=True)


def filter_front_matter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop proceedings front-matter (workshop intros, panel summaries, BoFs).
    These records have no abstract by construction and add noise to topic models.
    Signals: IEEE placeholder DOI (icse.YYYY.10xxx) or a workshop/panel title."""
    doi_hit = df["doi"].fillna("").str.contains(FRONT_MATTER_DOI)
    title_hit = df["title"].fillna("").str.contains(FRONT_MATTER_TITLE)
    drop = doi_hit | title_hit
    n_drop = int(drop.sum())
    print(f"Dropped {n_drop} front-matter records "
          f"({int(doi_hit.sum())} placeholder DOI, "
          f"{int(title_hit.sum())} title pattern, "
          f"{int((doi_hit & title_hit).sum())} doi+title overlap)")
    # Full list so an aggressive filter remains auditable in the run log.
    for title in df.loc[drop, "title"]:
        print(f"    {title[:100]}")
    return df[~drop].reset_index(drop=True)


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
    df = filter_to_papers(df)
    df = filter_main_track(df)
    df = filter_front_matter(df)
    df = deduplicate(df)
    df = flag_missing(df)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    out = INTERIM_DIR / f"{venue_key}_dblp.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    clean_venue("icse")