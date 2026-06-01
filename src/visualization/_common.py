"""
_common.py — shared loaders and output helpers for the interactive Plotly figures.

Every interactive visualization in this package reads the same three processed
tables and (for the impact views) joins per-paper citation counts from the
enriched silver layer. Centralising that here keeps each plot module focused on
the figure it draws rather than on data wrangling.

Each figure is written as a self-contained HTML in outputs/figures/ (plotly.js
pulled from CDN) — the interactive artifact is the whole point, so we don't
flatten these to static PDFs.
"""
from pathlib import Path

import pandas as pd

# Anchor to the repo root (this file lives at src/visualization/_common.py) so
# the figures can be generated from any working directory.
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data/processed"
INTERIM_DIR = ROOT / "data/interim"
FIGURES_DIR = ROOT / "outputs/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BUCKET_YEARS = 5


def short_label(top_words, n: int = 3) -> str:
    """First n top-words joined with ' · '. Accepts a list/array or a
    comma-joined string (the two tables store top_words differently)."""
    if top_words is None:
        return ""
    if isinstance(top_words, str):
        words = [w.strip() for w in top_words.split(",")]
    else:
        words = [str(w).strip() for w in top_words]
    return " · ".join(w for w in words[:n] if w)


def load_topics() -> pd.DataFrame:
    """Topic summaries: topic_id, size, label, top_words (top_words is a list)."""
    return pd.read_parquet(PROCESSED_DIR / "icse_topics.parquet")


def load_tot() -> pd.DataFrame:
    """Time series: topic_id, top_words, freq, year_bucket, share."""
    return pd.read_parquet(PROCESSED_DIR / "icse_topics_over_time.parquet")


def topic_labels() -> dict[int, str]:
    """{topic_id: 'word1 · word2 · word3'} for every non-outlier topic."""
    topics = load_topics()
    return {int(r["topic_id"]): short_label(r["top_words"])
            for _, r in topics.iterrows() if int(r["topic_id"]) != -1}


def load_paper_topics() -> pd.DataFrame:
    """Per-paper topic assignments joined to title + citation_count + authors.

    Returns columns: dblp_key, year, topic_id, topic_probability, title,
    citation_count (NaNs filled with 0), authors, year_bucket, paper_url.
    paper_url prefers the publisher/DOI link (`ee`) and falls back to the
    always-present DBLP record URL.
    """
    pt = pd.read_parquet(PROCESSED_DIR / "icse_paper_topics.parquet")
    cols = ["dblp_key", "title", "citation_count", "authors", "url", "ee"]
    en = pd.read_parquet(INTERIM_DIR / "icse_enriched.parquet")[cols]
    m = pt.merge(en, on="dblp_key", how="left")
    m["citation_count"] = m["citation_count"].fillna(0.0)
    m["year_bucket"] = (m["year"] // BUCKET_YEARS) * BUCKET_YEARS
    m["paper_url"] = m["ee"].fillna(m["url"])
    return m


def save(fig, name: str) -> None:
    """Write the interactive HTML for a figure (plotly.js via CDN)."""
    dest = FIGURES_DIR / f"{name}.html"
    fig.write_html(str(dest), include_plotlyjs="cdn")
    print(f"  wrote {dest}")