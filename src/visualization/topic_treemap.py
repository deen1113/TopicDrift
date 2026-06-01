"""
topic_treemap.py — Hierarchical decade → topic treemap with a click-to-list panel.

WHAT IT DOES
  Lays out the whole corpus as a treemap: the first level is the publication
  decade, the second level is the topic within that decade. Each tile's area is
  the number of papers and its colour is the mean citations per paper, so big
  tiles are where the field published most and warm tiles are where that work
  was most cited.

  Clicking a topic tile renders a panel beneath the chart that lists every paper
  in that topic-and-decade: the title is a clickable link to the paper (the
  publisher/DOI link, falling back to the DBLP record), followed by the authors,
  year and citation count, sorted most-cited first.

WHAT IT HOPES TO ANSWER
  How is each era's research composed, which slices were high-impact — and,
  drilling in, exactly which papers make up any given topic in a given decade?

Reads:  data/processed/icse_paper_topics.parquet, data/interim/icse_enriched.parquet,
        data/processed/icse_topics.parquet
Writes: outputs/figures/topic_treemap.html
"""
import json
import re

import plotly.express as px

from _common import FIGURES_DIR, load_paper_topics, topic_labels

NAME = "topic_treemap"


def _clean_author(name) -> str:
    """Drop DBLP disambiguation suffixes, e.g. 'Michael Hicks 0001' → 'Michael Hicks'."""
    return re.sub(r"\s+\d{3,}$", "", str(name)).strip()


def _authors_str(arr) -> str:
    names = [_clean_author(a) for a in (list(arr) if arr is not None else [])]
    if not names:
        return ""
    if len(names) > 8:
        return ", ".join(names[:8]) + f", … (+{len(names) - 8})"
    return ", ".join(names)


def build():
    """Return (tile aggregate, {'<decade>||<topic>': [paper dicts]} for the panel)."""
    pt = load_paper_topics()
    labels = topic_labels()
    pt = pt[pt["topic_id"].isin(labels)].copy()
    pt["decade"] = (pt["year"] // 10 * 10).astype(int).astype(str) + "s"
    pt["topic"] = pt["topic_id"].map(labels)

    agg = pt.groupby(["decade", "topic"]).agg(
        papers=("dblp_key", "size"),
        mean_citations=("citation_count", "mean"),
    ).reset_index()

    papers: dict[str, list[dict]] = {}
    for (decade, topic), grp in pt.sort_values("citation_count", ascending=False).groupby(["decade", "topic"]):
        papers[f"{decade}||{topic}"] = [
            {"t": r.title or "(untitled)", "a": _authors_str(r.authors),
             "u": r.paper_url or "", "y": int(r.year), "c": int(r.citation_count)}
            for r in grp.itertuples()
        ]
    return agg, papers


# JS injected into the HTML: builds a list panel under the treemap and fills it
# when a topic tile is clicked. {plot_id} is substituted by Plotly (literal
# replace, so the JSON braces below are safe).
_POST_SCRIPT = """
var DATA = __DATA__;
var gd = document.getElementById("{plot_id}");
var panel = document.createElement("div");
panel.style.cssText = "font-family:sans-serif;max-width:1100px;margin:18px auto;padding:0 14px;color:#222";
panel.innerHTML = "<p style='color:#777'>Click a topic tile above to list its papers.</p>";
gd.parentNode.insertBefore(panel, gd.nextSibling);

function esc(s){ return (s+"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

gd.on("plotly_click", function(d){
  var p = d.points[0];
  var parts = (p.id || "").split("/");
  if (parts.length < 3) { return; }            // decade/root click: nothing to list
  var decade = parts[1];
  var topic = parts.slice(2).join("/");
  var rows = DATA[decade + "||" + topic] || [];
  var html = "<h2 style='margin-bottom:4px'>" + esc(topic) + "</h2>"
           + "<p style='color:#777;margin-top:0'>" + esc(decade) + " · " + rows.length + " papers</p><ol>";
  rows.forEach(function(r){
    var link = r.u ? "<a href='" + esc(r.u) + "' target='_blank' rel='noopener'>" + esc(r.t) + "</a>" : esc(r.t);
    html += "<li style='margin-bottom:8px'>" + link
          + " <span style='color:#999'>(" + r.y + ", " + r.c + " citations)</span>"
          + "<br><span style='color:#555;font-size:0.9em'>" + esc(r.a) + "</span></li>";
  });
  html += "</ol>";
  panel.innerHTML = html;
  panel.scrollIntoView({behavior:"smooth", block:"start"});
});
"""


def plot(agg, papers):
    fig = px.treemap(
        agg, path=[px.Constant("ICSE"), "decade", "topic"],
        values="papers", color="mean_citations",
        color_continuous_scale="Turbo",
        title="ICSE corpus — decade → topic (click a topic to list its papers)",
        template="plotly_white",
    )
    fig.update_traces(hovertemplate=(
        "<b>%{label}</b><br>papers: %{value}"
        "<br>mean citations: %{color:.1f}<extra></extra>"))
    fig.update_layout(coloraxis_colorbar=dict(title="Mean<br>citations"),
                      margin=dict(t=60, l=10, r=10, b=10), height=800)

    data_json = json.dumps(papers, ensure_ascii=False).replace("</", "<\\/")
    post = _POST_SCRIPT.replace("__DATA__", data_json)
    dest = FIGURES_DIR / f"{NAME}.html"
    fig.write_html(str(dest), include_plotlyjs="cdn", post_script=post)
    print(f"  wrote {dest}")


def main():
    agg, papers = build()
    print(f"Treemap: {len(agg)} decade×topic tiles, "
          f"{sum(len(v) for v in papers.values())} papers indexed for the panel")
    plot(agg, papers)


if __name__ == "__main__":
    main()