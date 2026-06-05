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

import logging
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data/processed"
INTERIM_DIR = ROOT / "data/interim"
FIGURES_DIR = ROOT / "outputs/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BUCKET_YEARS = 5

# ── Multi-conference scopes (shared by the scope-based figures) ────────────────
# A scope is a venue filter over the one global topic space in the conf_* tables.
SCOPE_TITLES = {"icse": "ICSE", "top10": "Top 10 Conferences", "all": "All Conferences"}


def load_scopes() -> dict:
    """scopes block from config/venues.yaml: {scope: [venue, …] | filter}."""
    return yaml.safe_load((ROOT / "config/venues.yaml").read_text()).get("scopes", {})


def scope_filter(pt: pd.DataFrame, scope: str, scopes: dict | None = None) -> pd.DataFrame:
    """Rows of a conf_* table belonging to `scope` (no filter for 'all')."""
    venues = (scopes or load_scopes()).get(scope)
    if scope == "all" or not isinstance(venues, list):
        return pt
    return pt[pt["conf"].isin(set(venues))]


def load_conf_paper_topics() -> pd.DataFrame:
    """Global per-paper assignments: dblp_key, conf, year, topic_id, group."""
    return pd.read_parquet(PROCESSED_DIR / "conf_paper_topics.parquet")


def conf_topic_labels() -> dict[int, str]:
    """{topic_id: llm_label} for the global fit."""
    t = pd.read_parquet(PROCESSED_DIR / "conf_topics.parquet")
    return dict(zip(t["topic_id"].astype(int), t["llm_label"].astype(str)))


def conf_group_registry() -> tuple[dict[str, str], list[str]]:
    """({group: colour}, [group order]) from conf_topic_groups.parquet."""
    reg = pd.read_parquet(PROCESSED_DIR / "conf_topic_groups.parquet").sort_values("order")
    colors = {str(r["group"]): str(r["color"]) for _, r in reg.iterrows()}
    return colors, reg["group"].astype(str).tolist()


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


def save(fig, name: str) -> None:
    """Write the interactive HTML for a figure (plotly.js via CDN)."""
    dest = FIGURES_DIR / f"{name}.html"
    fig.write_html(str(dest), include_plotlyjs="cdn")
    log.info("  wrote %s", dest)
