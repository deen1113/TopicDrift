"""
drift_streamgraph.py — Interactive stacked-area drift chart of ICSE topics.

WHAT IT DOES
  Draws the top-N topics (by total paper volume) as a stacked area over the
  5-year buckets, where each band is the topic's share of that bucket's papers.
  This is the interactive successor to the static matplotlib chart: every band
  carries a rich hover (topic label, share %, raw paper count), the legend
  toggles topics on/off on click, and a dropdown switches the y-axis between
  "share of bucket" and "absolute paper count".

WHAT IT HOPES TO ANSWER
  How has each major research theme's prevalence at ICSE risen and fallen over
  1975-2025? Which topics dominated which eras, and when did the hand-offs
  happen (e.g. one theme receding as another climbs)?

  Because only the top-N topics are drawn, the bands do not fill to 1.0 — the
  headroom is the combined share of the omitted long-tail topics and outliers.

Reads:  data/processed/icse_topics_over_time.parquet, icse_topics.parquet
Writes: outputs/figures/drift_streamgraph.html
"""
import plotly.graph_objects as go

from _common import load_topics, load_tot, save, short_label

TOP_N = 15
NAME = "drift_streamgraph"


def build(tot, topics):
    """Return (ordered topic_ids, labels, buckets, share-series, freq-series)."""
    ranked = topics[topics["topic_id"] != -1].sort_values("size", ascending=False)
    top_ids = [int(t) for t in ranked["topic_id"].head(TOP_N)]
    labels = {int(r["topic_id"]): short_label(r["top_words"]) for _, r in ranked.iterrows()}

    buckets = sorted(tot["year_bucket"].unique())
    share, freq = {}, {}
    for tid in top_ids:
        sub = tot[tot["topic_id"] == tid].set_index("year_bucket")
        share[tid] = [float(sub["share"].get(b, 0.0)) for b in buckets]
        freq[tid] = [int(sub["freq"].get(b, 0)) for b in buckets]
    return top_ids, labels, buckets, share, freq


def plot(top_ids, labels, buckets, share, freq):
    fig = go.Figure()
    for tid in top_ids:
        fig.add_trace(go.Scatter(
            x=buckets, y=share[tid], name=labels[tid],
            mode="lines", stackgroup="one", hoveron="points+fills",
            customdata=freq[tid],
            hovertemplate=(f"<b>{labels[tid]}</b><br>"
                           "bucket: %{x}s<br>share: %{y:.1%}<br>"
                           "papers: %{customdata}<extra></extra>"),
        ))

    # Dropdown: restyle every trace's y between share and raw count.
    share_y = [share[tid] for tid in top_ids]
    freq_y = [freq[tid] for tid in top_ids]
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown", x=1.01, y=1.12, xanchor="left",
            buttons=[
                dict(label="Share of bucket", method="update",
                     args=[{"y": share_y},
                           {"yaxis": {"title": "Share of papers in bucket", "tickformat": ".0%"}}]),
                dict(label="Paper count", method="update",
                     args=[{"y": freq_y},
                           {"yaxis": {"title": "Papers in bucket"}}]),
            ],
        )],
    )
    fig.update_layout(
        title=f"ICSE topic drift — top {TOP_N} topics by volume",
        xaxis=dict(title="5-year bucket", tickmode="array",
                   tickvals=buckets, ticktext=[f"{b}s" for b in buckets]),
        yaxis=dict(title="Share of papers in bucket", tickformat=".0%"),
        legend=dict(title="Topic (top words)", x=1.01, y=1.0),
        hovermode="x unified", template="plotly_white",
    )
    save(fig, NAME)


def main():
    tot, topics = load_tot(), load_topics()
    print(f"Loaded {len(tot)} time-series rows across {tot['topic_id'].nunique()} topics")
    plot(*build(tot, topics))


if __name__ == "__main__":
    main()