"""Text-cleaning utilities shared across ingest and visualization."""

import re
import unicodedata

LATEX = re.compile(r"\$[^$]*\$")
HTML = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")
_DBLP_SUFFIX = re.compile(r"\s+\d{3,}$")


def normalize(text: str | None) -> str:
    """Lowercase, drop LaTeX/HTML, collapse whitespace; '' for non-strings."""
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = HTML.sub(" ", LATEX.sub(" ", s))
    return WS.sub(" ", s.lower()).strip()


def clean_author(name) -> str:
    """Drop DBLP disambiguation suffixes, e.g. 'Michael Hicks 0001' → 'Michael Hicks'."""
    return _DBLP_SUFFIX.sub("", str(name)).strip()


def clean_markup(text) -> str:
    """Strip HTML/XML tags and collapse whitespace; preserves case."""
    if not text:
        return ""
    return WS.sub(" ", HTML.sub(" ", str(text))).strip()
