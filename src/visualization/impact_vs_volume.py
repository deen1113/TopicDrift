"""
impact_vs_volume.py — Topic impact-vs-volume scatter (the influential niches).

WHAT IT DOES
  Plots one bubble per topic: x = number of papers (volume, log scale), y = mean
  citations per paper (impact), bubble size = total citations the topic has
  accrued, colour = the topic's median publication year (so older vs. newer
  themes are distinguishable). Median lines split the plane into quadrants, and
  hovering a bubble reveals the topic's top words and a few sample paper titles.

WHAT IT HOPES TO ANSWER
  Where does the field spend its effort versus where does impact concentrate?
  The quadrants surface high-volume / high-impact mainstays, small-but-
  influential niches (low volume, high impact), and crowded low-impact areas —
  a one-glance map of which topics punch above their weight.

Reads:  data/processed/icse_paper_topics.parquet, data/interim/icse_enriched.parquet,
        data/processed/icse_topics.parquet
Writes: outputs/figures/impact_vs_volume.html
"""
import plotly.express as px

from _common import load_paper_topics, save, topic_labels

NAME = "impact_vs_volume"


def build():
    """One row per topic: volume, mean/total citations, median year, samples."""
    pt = load_paper_topics()
    labels = topic_labels()
    pt = pt[pt["topic_id"].isin(labels)].copy()

    def sample_titles(s):
        titles = [t for t in s.dropna().tolist() if t]
        return "<br>".join(f"• {t[:70]}" for t in titles[:3])

    agg = pt.groupby("topic_id").agg(
        volume=("dblp_key", "size"),
        mean_citations=("citation_count", "mean"),
        total_citations=("citation_count", "sum"),
        median_year=("year", "median"),
        samples=("title", sample_titles),
    ).reset_index()
    agg["topic"] = agg["topic_id"].map(labels)
    return agg


def plot(agg):
    fig = px.scatter(
        agg, x="volume", y="mean_citations", size="total_citations",
        color="median_year", log_x=True, size_max=60,
        color_continuous_scale="Turbo", hover_name="topic",
        custom_data=["samples", "total_citations"],
        labels={"volume": "Papers (volume, log scale)",
                "mean_citations": "Mean citations / paper (impact)",
                "median_year": "Median year"},
        title="ICSE topics — impact vs. volume",
        template="plotly_white",
    )
    fig.update_traces(hovertemplate=(
        "<b>%{hovertext}</b><br>papers: %{x}<br>mean citations: %{y:.1f}"
        "<br>total citations: %{customdata[1]:,.0f}<br><br>%{customdata[0]}<extra></extra>"))
    fig.add_hline(y=agg["mean_citations"].median(), line_dash="dot", opacity=0.4)
    fig.add_vline(x=agg["volume"].median(), line_dash="dot", opacity=0.4)
    save(fig, NAME)


def main():
    agg = build()
    print(f"Impact scatter: {len(agg)} topics")
    plot(agg)


if __name__ == "__main__":
    main()