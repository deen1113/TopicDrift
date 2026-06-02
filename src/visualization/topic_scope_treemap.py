"""
topic_scope_treemap.py — Composition treemap per scope (decade → theme → topic).

A light, citation-free treemap that works identically for every scope
(ICSE / Top-10 / All), built straight from the global assignment table. Tile
area = #papers; colour = overarching theme. It answers "what is each era made
of, and how big is each theme/topic" — complementing the streamgraph's drift
view. (The richer citation/impact treemap stays ICSE-only in topic_treemap.py.)

Reads:
  data/processed/conf_paper_topics.parquet   dblp_key, conf, year, topic_id, group
  data/processed/conf_topics.parquet         topic_id → llm_label
  data/processed/conf_topic_groups.parquet   theme colour/order

Writes: outputs/figures/topic_scope_treemap_{scope}.html  (one per scope)
"""
from __future__ import annotations

import plotly.express as px

from _common import (
    FIGURES_DIR, SCOPE_TITLES, conf_group_registry, conf_topic_labels,
    load_conf_paper_topics, load_scopes, scope_filter,
)

NAME = "topic_scope_treemap"


def write_scope(scope, pt_scope, id_to_label, color):
    df = pt_scope.dropna(subset=["group"]).copy()
    df["decade"] = (df["year"] // 10 * 10).astype(int).astype(str) + "s"
    df["theme"] = df["group"].astype(str)
    df["topic"] = df["topic_id"].astype(int).map(id_to_label)

    agg = (df.groupby(["decade", "theme", "topic"]).size()
             .rename("papers").reset_index())
    title = SCOPE_TITLES.get(scope, scope)
    fig = px.treemap(
        agg, path=[px.Constant(title), "decade", "theme", "topic"],
        values="papers", color="theme", color_discrete_map=color,
        title=f"Corpus composition — {title} (decade → theme → topic)",
        template="plotly_white",
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>papers: %{value}<extra></extra>",
                      root_color="lightgrey")
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10), height=760, showlegend=False)
    dest = FIGURES_DIR / f"{NAME}_{scope}.html"
    fig.write_html(str(dest), include_plotlyjs="cdn")
    print(f"  wrote {dest.name} ({len(agg)} tiles, {len(pt_scope):,} papers)")


def main():
    id_to_label = conf_topic_labels()
    color, _ = conf_group_registry()
    pt = load_conf_paper_topics()
    scopes = load_scopes()
    for scope in SCOPE_TITLES:
        s = scope_filter(pt, scope, scopes)
        if len(s):
            write_scope(scope, s, id_to_label, color)


if __name__ == "__main__":
    main()
