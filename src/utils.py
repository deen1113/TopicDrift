"""Shared helpers used by every pipeline stage.

Centralises filesystem paths, the venue/stopword config loaders, and a small
logger factory so the other modules don't reinvent any of this.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

# All paths are anchored at the project root (the parent of src/), so the
# pipeline works no matter what directory you launch it from.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_FIGURES = ROOT / "outputs" / "figures"
OUTPUTS_TABLES = ROOT / "outputs" / "tables"


def load_venues() -> dict:
    """Return the parsed config/venues.yaml as {venue_key: {dblp_slugs, start_year, end_year}}."""
    return yaml.safe_load((CONFIG_DIR / "venues.yaml").read_text())


def load_stopwords() -> set[str]:
    """Return the curated stopword list, ignoring blanks and '#'-prefixed comments."""
    words = set()
    for line in (CONFIG_DIR / "stopwords.txt").read_text().splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    return words


def setup_logging(name: str) -> logging.Logger:
    """One-line stdout logger; safe to call repeatedly (won't double-attach handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s",
                                          datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, OUTPUTS_FIGURES, OUTPUTS_TABLES):
        d.mkdir(parents=True, exist_ok=True)


def slug_to_filename(slug: str) -> str:
    """DBLP slugs contain '/'; flatten so we can use them as filename stems."""
    return slug.replace("/", "_")


def env(key: str) -> str | None:
    """Tiny os.environ.get wrapper so callers don't import os directly."""
    return os.environ.get(key)
