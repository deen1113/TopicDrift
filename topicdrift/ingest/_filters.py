"""
Shared post-fetch filters for the per-venue silver pipeline.

The DBLP dump (ingest_dblp_dump.py) already drops corrigenda/datasets via
publtype and restricts to `conf/*` keys. Once we slice it down to a single
venue, we still need to:
  1. drop proceedings front-matter (workshop intros, panel summaries, BoFs)
     that have no abstract by construction;
  2. deduplicate by dblp_key and title+year (occasional DBLP duplicates);
  3. add a has_doi flag for downstream enrichment.

These run on the per-venue slice, not the whole dump, so cost is negligible.
"""
import re

import pandas as pd

# Title-shape signals for proceedings front-matter. Audited against ICSE,
# every match was a non-research entry; should generalize to other venues.
FRONT_MATTER_TITLE = re.compile(
    r"(?:"
    r"^(?:the\s+)?\d+(?:st|nd|rd|th)\s+(?:international\s+|[a-z]+\s+)?workshop\b"
    r"|^(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
    r"\s+(?:international\s+|[a-z]+\s+)?workshop\b"
    r"|^(?:international\s+)?workshop\s+(?:on|to)\b"
    r"|^[a-z]+\s+workshop\b"
    r"|^[a-z]+[\s-]?\d{1,4}\s*[:\-].*\bworkshop\b"
    r"|^bof:"
    r"|^panel:"
    r"|^tutorial:"
    r"|^keynote:"
    r"|^the\s+international\s+symposium\b"
    r"|^message from\b"
    r"|\bpanel\s+summary\b"
    r"|\(panel[^)]*\)\s*$"
    r"|\(tutorial[^)]*\)\s*$"
    r"|\(workshop\s+report\)"
    r"|\(workshop\s+session\)"
    r"|\bfaculty\s+symposium\b"
    r"|\bdoctoral\s+symposium\b"
    r"|\bphd\s+symposium\b"
    r")",
    re.I,
)


def front_matter_doi_re(acronym: str) -> re.Pattern[str]:
    """IEEE assigns 5-digit DOIs ending in 10xxx to proceedings front-matter
    (workshop intros, panels, BoFs) — same shape across venues."""
    return re.compile(rf"{re.escape(acronym)}\.\d{{4}}\.10\d{{3}}$", re.I)


def filter_front_matter(df: pd.DataFrame, acronym: str) -> pd.DataFrame:
    """Drop proceedings front-matter rows (matched by DOI or title)."""
    doi_hit = df["doi"].fillna("").str.contains(front_matter_doi_re(acronym))
    title_hit = df["title"].fillna("").str.contains(FRONT_MATTER_TITLE)
    drop = doi_hit | title_hit
    n_drop = int(drop.sum())
    print(f"  dropped {n_drop} front-matter rows "
          f"({int(doi_hit.sum())} DOI, {int(title_hit.sum())} title)")
    return df[~drop].reset_index(drop=True)


def deduplicate(df: pd.DataFrame, dupes_path=None) -> pd.DataFrame:
    """Deduplicate by dblp_key then title+year. Prefers rows with DOIs and
    more authors when collapsing title+year duplicates."""
    before = len(df)
    df = df.drop_duplicates(subset=["dblp_key"], keep="first")

    title_dupes = df[df.duplicated(subset=["title", "year"], keep=False)]
    if not title_dupes.empty and dupes_path is not None:
        title_dupes.sort_values(["year", "title"]).to_csv(dupes_path, index=False)
        print(f"  wrote {len(title_dupes)} duplicate rows to {dupes_path}")

    n_authors = df["authors"].map(lambda xs: len(xs) if xs is not None else 0)
    df = df.assign(
        _has_doi=df["doi"].notna(),
        _n_authors=n_authors,
    ).sort_values(["_has_doi", "_n_authors"], ascending=[False, False])
    df = df.drop_duplicates(subset=["title", "year"], keep="first")
    df = df.drop(columns=["_has_doi", "_n_authors"]).reset_index(drop=True)
    print(f"  deduplicated: {before} -> {len(df)} rows")
    return df


def flag_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Add has_doi. Papers without DOI can't be DOI-matched against OpenAlex."""
    df = df.copy()
    df["has_doi"] = df["doi"].notna()
    missing = int((~df["has_doi"]).sum())
    if len(df):
        print(f"  papers missing DOI: {missing} ({100 * missing / len(df):.1f}%)")
    return df
