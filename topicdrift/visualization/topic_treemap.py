"""
topic_treemap.py — Hierarchical decade → topic treemap with a click-to-list panel.

WHAT IT DOES
  Lays out a scope's corpus as a treemap: root → decade → overarching theme →
  topic. Each tile's area is the number of papers.

  Colour is the overarching theme: every subtopic takes its theme's colour, so a
  theme reads as one coloured block (tile text auto-set to black/white for
  contrast). Mode toggles re-colour the topic tiles by median citations (an
  impact heatmap), growth-vs-previous-decade, or whether a topic is new this
  decade. Hover shows the key stats; clicking a topic tile lists its papers below.

  THREE SCOPES, TWO DATA SOURCES
    • icse  — the ICSE-only fit (icse_* tables). Fully enriched: citation counts,
              authors and paper URLs are present, so every colour mode and every
              stat card is available. This is the original, richest treemap.
    • top10 / all — the global multi-conference fit (conf_* tables). These have
              titles + abstracts but NO citation/author/URL enrichment, so the
              impact colour mode, the citation stat cards, author lists and paper
              links are dropped. Identity / Growth / Emerging modes and a
              title+year paper list all still work (they are share-based).

  Everything that needs citations/authors/links is guarded by `has_cites` on
  both the Python and the JS side, degrading gracefully when the data is absent.

Reads (icse):  data/processed/icse_paper_topics.parquet (+ icse_enriched join),
               data/processed/icse_topics.parquet
Reads (conf):  data/processed/conf_paper_topics.parquet, conf_enriched.parquet,
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
        agg, papers, meta, group_colors, has_cites = build(scope)
        log.info(
            "%s: %d decade×theme×topic tiles, %d theme tiles, "
            "%d papers indexed, %d decade descriptions%s",
            SCOPE_TITLES[scope],
            len(agg),
            len(meta["themes"]),
            sum(len(v["rows"]) for v in papers.values()),
            len(meta["decades"]),
            "" if has_cites else " (no citation data — reduced)",
        )
        plot(scope, agg, papers, meta, group_colors, has_cites)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
