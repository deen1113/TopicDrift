"""
venue.py — Slice the DBLP-wide dump down to one (or more) venues.

Reads:  data/interim/dblp_conf.parquet     (from ingest_dblp_dump.py)
Writes: data/interim/<venue>_dblp.parquet  (per venue)

Replaces the old ingest.py + clean.py pair. There's no per-venue DBLP API fetch
anymore — `make dump` (which produces dblp_conf.parquet) is a prerequisite.

By default a venue-agnostic main-track filter runs over the `booktitle` column
(`is_main_track` in _filters), dropping companion volumes, workshop summaries,
and standalone satellite events filed under the same DBLP slug. Pass
`--include-companion` to keep everything (Workflow B already does this; you
only want the broad set for analysis that filters by abstract coverage later).

The downstream enrich_openalex.py reads <venue>_dblp.parquet, so this writes
the same column set it expects:
  dblp_key, title, year, doi, authors, ee, url, venue, has_doi
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from topicdrift.ingest._filters import (
    deduplicate,
    filter_front_matter,
    filter_main_track,
    flag_missing,
)

log = logging.getLogger(__name__)

INTERIM_DIR = Path("data/interim")
DUMP_PATH = INTERIM_DIR / "dblp_conf.parquet"


def slice_venue(
    venue_key: str, dump: pd.DataFrame, include_companion: bool = False
) -> pd.DataFrame:
    """Filter the dump to one venue (by `conf/<venue_key>`), apply shared
    filters, return the per-venue silver DataFrame."""
    slug = f"conf/{venue_key}"
    df = dump[dump["conf"] == slug].copy()
    log.info("[%s] %d rows in dump for %s", venue_key, len(df), slug)
    if df.empty:
        log.warning("  WARNING: no rows for %s. Check the DBLP slug spelling.", slug)
        return df

    if not include_companion:
        df = filter_main_track(df, slug_acronym=venue_key)
    df = filter_front_matter(df, acronym=venue_key)
    df = deduplicate(df, dupes_path=INTERIM_DIR / f"{venue_key}_duplicates.csv")
    df = flag_missing(df)
    df = df.dropna(subset=["year"]).reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["venue"] = venue_key
    return df


def build_venue(venue_key: str, include_companion: bool = False) -> None:
    if not DUMP_PATH.exists():
        raise SystemExit(f"Missing {DUMP_PATH}. Run `make dump` first to build the DBLP dump.")
    dump = pd.read_parquet(DUMP_PATH)
    df = slice_venue(venue_key, dump, include_companion=include_companion)
    out = INTERIM_DIR / f"{venue_key}_dblp.parquet"
    df.to_parquet(out, index=False)
    log.info("[%s] wrote %s (%d rows)", venue_key, out, len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = sys.argv[1:]
    include_companion = "--include-companion" in args
    venues = [a for a in args if not a.startswith("--")] or ["icse"]
    for v in venues:
        build_venue(v, include_companion=include_companion)
