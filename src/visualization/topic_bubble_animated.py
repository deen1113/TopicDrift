"""
topic_bubble_animated.py — Prevalence-vs-impact bubble motion chart over time.

WHAT IT DOES
  A Hans-Rosling-style animated bubble chart. Each bubble is a topic in one
  5-year bucket: x = mean citations of that topic's papers in that bucket
  (impact), y = the topic's share of the bucket (prevalence), bubble size =
  number of papers, colour = topic. A play button (and slider) walks the field
  forward from 1975 to 2025 so you can watch topics swell, drift right as they
  accrue citations, and shrink away.

WHAT IT HOPES TO ANSWER
  Which topics are merely popular versus genuinely influential, and how does
  that relationship move across decades? Do high-volume topics also attract
  citations, or does impact concentrate in smaller, newer themes?

Reads:  data/processed/icse_topics_over_time.parquet,
        data/processed/icse_paper_topics.parquet, data/interim/icse_enriched.parquet
Writes: outputs/figures/topic_bubble_animated.html
"""
import plotly.express as px

from _common import load_paper_topics, load_tot, save, topic_labels

NAME = "topic_bubble_animated"
MIN_PAPERS = 3  # hide topic/bucket cells with too few papers to be meaningful


def build():
    """One row per (topic, bucket): share, freq, mean citations, label."""
    tot = load_tot()
    pt = load_paper_topics()
    labels = topic_labels()

    impact = (pt[pt["topic_id"] != -1]
              .groupby(["topic_id", "year_bucket"])["citation_count"]
              .mean().reset_index(name="mean_citations"))

    df = tot.merge(impact, on=["topic_id", "year_bucket"], how="inner")
    df = df[df["topic_id"].isin(labels) & (df["freq"] >= MIN_PAPERS)].copy()
    df["topic"] = df["topic_id"].map(labels)
    df["bucket"] = df["year_bucket"].astype(int).astype(str) + "s"
    return df.sort_values("year_bucket")


def plot(df):
    xmax = df["mean_citations"].max() * 1.05
    ymax = df["share"].max() * 1.10

    animated = px.scatter(
        df, x="mean_citations", y="share", size="freq", color="topic",
        animation_frame="bucket", animation_group="topic",
        hover_name="topic", size_max=55,
        range_x=[0, xmax], range_y=[0, ymax],
        labels={"mean_citations": "Mean citations / paper (impact)",
                "share": "Share of bucket (prevalence)"},
        title="ICSE topics — prevalence vs. impact over time",
        template="plotly_white",
    )
    animated.update_yaxes(tickformat=".0%")
    animated.update_layout(showlegend=False)
    save(animated, NAME)


def main():
    df = build()
    print(f"Bubble chart: {len(df)} topic×bucket cells (≥{MIN_PAPERS} papers each)")
    plot(df)


if __name__ == "__main__":
    main()