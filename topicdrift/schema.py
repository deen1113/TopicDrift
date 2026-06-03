"""
schema.py — Expected column sets for the project's parquet files.

Call validate(df, SCHEMA_NAME) after loading a parquet to get an early,
readable error when a required column is absent rather than a confusing
KeyError deep in the pipeline.
"""

import pandas as pd


def validate(df: pd.DataFrame, required: list[str], name: str = "DataFrame") -> pd.DataFrame:
    """Raise ValueError if any required column is absent. Returns df unchanged."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    return df


# ── Interim parquets (ingest layer) ───────────────────────────────────────────

DBLP_CONF = ["conf", "dblp_key", "title", "year", "doi", "authors", "ee", "url", "booktitle"]

VENUE_DBLP = ["dblp_key", "title", "year", "doi", "authors", "ee", "url", "venue", "has_doi"]

VENUE_ENRICHED = [
    "dblp_key",
    "title",
    "year",
    "doi",
    "venue",
    "abstract",
    "has_abstract",
    "text",
    "oa_concepts",
    "citation_count",
    "openalex_id",
    "oa_type",
]

CONF_ENRICHED = ["conf", "dblp_key", "title", "year", "doi", "abstract", "has_abstract", "text"]

# ── Processed parquets (analysis layer) ───────────────────────────────────────

ICSE_TOPICS = ["topic_id", "size", "label", "top_words"]

ICSE_PAPER_TOPICS = ["dblp_key", "year", "topic_id", "topic_probability"]

ICSE_TOPICS_OVER_TIME = ["topic_id", "top_words", "freq", "year_bucket", "share"]

CONF_TOPICS = ["topic_id", "top_words", "llm_label", "size"]

CONF_PAPER_TOPICS = ["dblp_key", "conf", "year", "topic_id"]
