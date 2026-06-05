"""
topic_treemap.py — Hierarchical decade → topic treemap with a click-to-list panel.

WHAT IT DOES
  Lays out a scope's corpus as a treemap: root → decade → overarching theme →
  topic. Each tile's area is the number of papers.

  Colour is the overarching theme: every subtopic takes its theme's colour, so a
  theme reads as one coloured block (tile text auto-set to black/white for
  contrast). Mode toggles re-colour the topic tiles by growth-vs-previous-decade
  or whether a topic is new this decade. Hover shows the key stats; clicking a
  topic tile lists its papers below.

  THREE SCOPES, all using the global multi-conference fit (conf_* tables):
    • icse  — ICSE papers only
    • top10 — ten flagship SE/PL venues
    • all   — every venue with usable abstracts (2,000+ venues)

Reads: data/processed/conf_paper_topics.parquet, conf_enriched.parquet,
       conf_topics.parquet
Writes: outputs/figures/topic_treemap_{scope}.html  (one per scope)
"""

import logging

from topicdrift.visualization._common import SCOPE_TITLES
from topicdrift.visualization._treemap_data import build
from topicdrift.visualization._treemap_layout import plot

log = logging.getLogger(__name__)


def main():
    for scope in SCOPE_TITLES:
        agg, papers, meta, group_colors = build(scope)
        log.info(
            "%s: %d decade×theme×topic tiles, %d theme tiles, "
            "%d papers indexed, %d decade descriptions",
            SCOPE_TITLES[scope],
            len(agg),
            len(meta["themes"]),
            sum(len(v["rows"]) for v in papers.values()),
            len(meta["decades"]),
        )
        plot(scope, agg, papers, meta, group_colors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
