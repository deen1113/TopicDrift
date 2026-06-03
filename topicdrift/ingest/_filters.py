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

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

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
    log.info(
        "  dropped %d front-matter rows (%d DOI, %d title)",
        n_drop,
        int(doi_hit.sum()),
        int(title_hit.sum()),
    )
    return df[~drop].reset_index(drop=True)


def deduplicate(df: pd.DataFrame, dupes_path=None) -> pd.DataFrame:
    """Deduplicate by dblp_key then title+year. Prefers rows with DOIs and
    more authors when collapsing title+year duplicates."""
    before = len(df)
    df = df.drop_duplicates(subset=["dblp_key"], keep="first")

    title_dupes = df[df.duplicated(subset=["title", "year"], keep=False)]
    if not title_dupes.empty and dupes_path is not None:
        title_dupes.sort_values(["year", "title"]).to_csv(dupes_path, index=False)
        log.info("  wrote %d duplicate rows to %s", len(title_dupes), dupes_path)

    n_authors = df["authors"].map(lambda xs: len(xs) if xs is not None else 0)
    df = df.assign(
        _has_doi=df["doi"].notna(),
        _n_authors=n_authors,
    ).sort_values(["_has_doi", "_n_authors"], ascending=[False, False])
    df = df.drop_duplicates(subset=["title", "year"], keep="first")
    df = df.drop(columns=["_has_doi", "_n_authors"]).reset_index(drop=True)
    log.info("  deduplicated: %d -> %d rows", before, len(df))
    return df


def flag_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Add has_doi. Papers without DOI can't be DOI-matched against OpenAlex."""
    df = df.copy()
    df["has_doi"] = df["doi"].notna()
    missing = int((~df["has_doi"]).sum())
    if len(df):
        log.info("  papers missing DOI: %d (%.1f%%)", missing, 100 * missing / len(df))
    return df


def is_main_track(booktitle: str | None, acronym: str) -> bool:
    """Venue-agnostic main-track test against the DBLP <booktitle>.

    Main-track booktitles start with the venue acronym and continue with
    end-of-string, a space, a dash, or an opening paren (covers split
    volumes like "ICSE (1)" and colocated research tracks like "ICSE (SEIP)",
    "ICSE-NIER"). Companion volumes and workshop summaries are excluded by
    the "companion" / "workshop" substring guard.

    Standalone workshop booktitles (e.g. "CHASE", "SEAMS", "AST@ICSE") don't
    start with the acronym and are dropped automatically.
    """
    if not isinstance(booktitle, str) or not booktitle.strip():
        return False
    bt = booktitle.strip()
    a = acronym.upper()
    btu = bt.upper()
    if btu == a:
        return True
    if not (btu.startswith(f"{a} ") or btu.startswith(f"{a}-") or btu.startswith(f"{a}(")):
        return False
    blow = bt.lower()
    return "companion" not in blow and "workshop" not in blow


def infer_acronym(booktitles: pd.Series) -> str | None:
    """Guess a venue's canonical acronym from its booktitle distribution.

    Picks the most-common booktitle that doesn't look like a workshop/companion
    entry, then keeps the head (everything before " (" or "-"). Handles venues
    whose DBLP slug differs from the booktitle, e.g. conf/kbse → "ASE".
    """
    bt = booktitles.dropna().astype(str)
    bt = bt[~bt.str.contains(r"@|companion|workshop", case=False, regex=True, na=False)]
    if bt.empty:
        return None
    top = bt.value_counts().idxmax()
    head = re.split(r"\s*\(", str(top), maxsplit=1)[0]
    head = re.split(r"-", head, maxsplit=1)[0]
    return head.strip().upper() or None


def filter_main_track(df: pd.DataFrame, slug_acronym: str) -> pd.DataFrame:
    """Keep only main-track + colocated-research-track rows (drop companion,
    workshop summaries, and standalone satellite events filed under the same
    DBLP slug). Requires a `booktitle` column from the DBLP dump.

    Matches both the slug-derived acronym and the inferred-from-data acronym
    (so renamed venues like KBSE → ASE keep both eras of papers).
    """
    if "booktitle" not in df.columns:
        log.warning("  WARNING: booktitle column missing — main-track filter skipped")
        return df
    inferred = infer_acronym(df["booktitle"])
    acronyms = {slug_acronym.upper()} | ({inferred} if inferred else set())
    note = f"acronyms={'/'.join(sorted(acronyms))}"
    keep = df["booktitle"].map(lambda bt: any(is_main_track(bt, a) for a in acronyms))
    n_drop = int((~keep).sum())
    log.info("  dropped %d non-main-track rows (%s)", n_drop, note)
    return df[keep].reset_index(drop=True)
