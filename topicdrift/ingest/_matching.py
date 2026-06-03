"""Title-matching helpers shared by ingest modules.

Extracted here so enrich_openalex.py and enrich_acm.py can both use them
without either importing privates from the other.
"""

import re

import pandas as pd

from topicdrift.utils.text import WS, normalize

# DBLP often prepends "On"/"On a"/"On the" to titles that OpenAlex stores
# without. Strip these (and bare articles) so the loose matcher can equate them.
LEADING_ARTICLES = re.compile(r"^(?:on\s+(?:a\s+|an\s+|the\s+)?|a\s+|an\s+|the\s+)+")


def _loose_key(text: str | None) -> str:
    """normalize(), strip punctuation, then strip leading articles so trivially-
    different titles compare equal (DBLP-style "On"/"On the" prefixes)."""
    s = WS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", normalize(text))).strip()
    return LEADING_ARTICLES.sub("", s)


def _strict_title_match(title: str, year: int, work: dict) -> bool:
    if work.get("publication_year") != year:
        return False
    work_title = work.get("display_name") or ""
    return normalize(work_title) == normalize(title)


def _loose_title_match(title: str, year: int, work: dict) -> bool:
    """Punctuation-insensitive title equality within +/-1 year (recovers year
    drift and trailing-punctuation differences that strict matching rejects)."""
    work_year = work.get("publication_year")
    if work_year is None or abs(work_year - year) > 1:
        return False
    return _loose_key(work.get("display_name")) == _loose_key(title)


def _recompute_text_fields(out: pd.DataFrame) -> None:
    out["has_abstract"] = out["abstract"].fillna("").str.len() > 0
    out["text"] = (out["title"].map(normalize) + " " + out["abstract"].map(normalize)).str.strip()
