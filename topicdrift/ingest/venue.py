"""
venue.py — Slice the DBLP-wide dump down to one (or more) venues.

Reads:  data/interim/dblp_conf.parquet     (from ingest_dblp_dump.py)
Writes: data/interim/<venue>_dblp.parquet  (per venue)

Replaces the old ingest.py + clean.py pair. There's no per-venue DBLP API fetch
anymore — `make dump` (which produces dblp_conf.parquet) is a prerequisite.

The downstream enrich_openalex.py reads <venue>_dblp.parquet, so this writes
the same column set it expects:
  dblp_key, title, year, doi, authors, ee, url, venue, has_doi
"""

import sys
from pathlib import Path

import pandas as pd

from topicdrift.ingest._filters import deduplicate, filter_front_matter, flag_missing

INTERIM_DIR = Path("data/interim")
DUMP_PATH = INTERIM_DIR / "dblp_conf.parquet"


def slice_venue(venue_key: str, dump: pd.DataFrame) -> pd.DataFrame:
    """Filter the dump to one venue (by `conf/<venue_key>`), apply shared
    filters, return the per-venue silver DataFrame."""
    slug = f"conf/{venue_key}"
    df = dump[dump["conf"] == slug].copy()
    print(f"[{venue_key}] {len(df)} rows in dump for {slug}")
    if df.empty:
        print(f"  WARNING: no rows for {slug}. Check the DBLP slug spelling.")
        return df

    df = filter_front_matter(df, acronym=venue_key)
    df = deduplicate(df, dupes_path=INTERIM_DIR / f"{venue_key}_duplicates.csv")
    df = flag_missing(df)
    df = df.dropna(subset=["year"]).reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["venue"] = venue_key
    return df


def build_venue(venue_key: str) -> None:
    if not DUMP_PATH.exists():
        raise SystemExit(
            f"Missing {DUMP_PATH}. Run `make dump` first to build the DBLP dump."
        )
    dump = pd.read_parquet(DUMP_PATH)
    df = slice_venue(venue_key, dump)
    out = INTERIM_DIR / f"{venue_key}_dblp.parquet"
    df.to_parquet(out, index=False)
    print(f"[{venue_key}] wrote {out} ({len(df)} rows)\n")


if __name__ == "__main__":
    venues = sys.argv[1:] or ["icse"]
    for v in venues:
        build_venue(v)
