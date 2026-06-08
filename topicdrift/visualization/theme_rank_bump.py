"""
theme_rank_bump.py — the shifting pecking order of ICSE research, as continuous
rank-over-time trend lines for the 10 themes.

WHAT IT DOES
  The timeline is sampled in 3-year periods; in each, the 10 themes are ranked by
  their share of that period's papers (rank 1 = most-studied). Those period ranks
  are connected with smooth lines — no markers — so each theme reads as one
  continuous curve gliding up and down the ranks over the field's history. Where
  two lines cross, one theme overtook another. Themes are identified by colour in
  the legend (not on the axis), so the rank scale stays clean.

      y (1 at top) = the theme's rank by share of papers in the 3-year period
      line cross   = one theme overtaking another
      colour       = theme (see legend)

WHAT IT HOPES TO ANSWER
  The streamgraph shows how big each theme is; this shows the *order* — who leads,
  who slipped, and when the hierarchy reshuffled — as one continuous story rather
  than a stack of magnitudes.

Reads:  data/processed/icse_paper_topics.parquet, data/processed/icse_topics.parquet,
        data/processed/conf_topics.parquet (theme colours)  (via _common paths)
Writes: outputs/figures/theme_rank_bump.html
"""

import plotly.express as px
import plotly.graph_objects as go

from topicdrift.visualization._common import load_paper_topics, load_topics, save

NAME = "theme_rank_bump"
BUCKET = 5  # sampling period in years (one dot per period)
# A distinct categorical palette (not the canonical theme colours, which include
# near-duplicate blues/oranges) so 10 crossing lines stay tellable apart. Drop
# Dark24's near-black (#222A2A), which would disappear on the dark-mode canvas.
PALETTE = [c for c in px.colors.qualitative.Dark24 if c != "#222A2A"]

# JS injected after the plot: a dark-mode toggle (top-right) that recolours the
# whole figure; also obeys a parent page's `td-dark` postMessage when embedded.
_POST_SCRIPT = """
var gd = document.getElementById("{plot_id}");
var style = document.createElement("style");
style.textContent =
  "body{transition:background .2s,color .2s}"
+ "body.dark{background:#1e1e1e;color:#e6e6e6}"
+ "#td-dm-wrap{color:#3a3f4a;font-family:Inter,sans-serif}"
+ "body.dark #td-dm-wrap{color:#cfd3d9}";
document.head.appendChild(style);

function applyDark(on){
  document.body.classList.toggle("dark", !!on);
  Plotly.relayout(gd, on ? {
    paper_bgcolor:"#1e1e1e", plot_bgcolor:"#1e1e1e", "font.color":"#e6e6e6",
    "title.font.color":"#f0f0f0",
    "legend.bgcolor":"rgba(40,40,40,0.85)", "legend.bordercolor":"#555",
    "legend.title.font.color":"#e6e6e6"
  } : {
    paper_bgcolor:"white", plot_bgcolor:"white", "font.color":"#3a3f4a",
    "title.font.color":"#23272e",
    "legend.bgcolor":"rgba(255,255,255,0.8)", "legend.bordercolor":"#e2e4ea",
    "legend.title.font.color":"#3a3f4a"
  });
}
if (window.self === window.top){
  var dm = document.createElement("div");
  dm.id = "td-dm-wrap";
  dm.style.cssText = "position:absolute;top:12px;right:18px;z-index:1000;font-size:12px;display:flex;align-items:center;gap:7px";
  dm.innerHTML = "<span>Dark mode</span><label style='position:relative;display:inline-block;width:40px;height:21px'>"
    + "<input type='checkbox' id='td-dm' style='opacity:0;width:0;height:0'>"
    + "<span id='td-sw' style='position:absolute;cursor:pointer;inset:0;background:#ccc;border-radius:22px;transition:.2s'>"
    + "<span id='td-knob' style='position:absolute;height:15px;width:15px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.2s'></span></span></label>";
  gd.style.position = "relative";
  gd.appendChild(dm);
  document.getElementById("td-dm").addEventListener("change", function(e){
    var on = e.target.checked; applyDark(on);
    document.getElementById("td-sw").style.background = on ? "#4a90d9" : "#ccc";
    document.getElementById("td-knob").style.transform = on ? "translateX(19px)" : "";
  });
}
window.addEventListener("message", function(ev){
  if (ev.data && ev.data.type === "td-dark"){ applyDark(ev.data.on); }
});
"""


def build():
    """Return the period centres, period labels, per-theme {rank, share}
    trajectories, the theme labels and their colours."""
    pt = load_paper_topics()
    t = load_topics()
    label = dict(zip(t["topic_id"].astype(int), t["label"].astype(str)))
    sizes = t.set_index("topic_id")["size"]
    color = {tid: PALETTE[i % len(PALETTE)] for i, tid in enumerate(sorted(label))}

    pt = pt.assign(bucket=(pt["year"] // BUCKET) * BUCKET)
    buckets = sorted(pt["bucket"].unique())
    counts = (pt.groupby(["topic_id", "bucket"]).size()
              .unstack("bucket").reindex(columns=buckets).fillna(0.0))
    # order rows by total size (desc) so rank ties — e.g. the all-zero early years —
    # break consistently toward the larger theme instead of jumping around
    counts = counts.reindex(sizes.sort_values(ascending=False).index)

    share = counts.div(counts.sum(axis=0), axis=1)
    rank = share.rank(axis=0, ascending=False, method="first")

    centres = [float(b) for b in buckets]  # one dot per period, at its start year
    periods = [f"{b}–{b + BUCKET - 1}" for b in buckets]
    series = {int(tid): {"rank": rank.loc[tid].tolist(), "share": share.loc[tid].tolist()}
              for tid in counts.index}
    return centres, periods, series, label, color


def plot(centres, periods, series, label, color):
    FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
    n = len(series)
    fig = go.Figure()

    # draw in current-rank order so the legend reads top → bottom like the chart
    final_rank = {tid: s["rank"][-1] for tid, s in series.items()}
    for tid in sorted(series, key=lambda t: final_rank[t]):
        s = series[tid]
        fig.add_trace(go.Scatter(
            x=centres, y=s["rank"], mode="lines+markers", name=label[tid],
            line=dict(color=color[tid], width=2.2, shape="linear"),
            marker=dict(color=color[tid], size=6, line=dict(width=0)),
            customdata=[[p, sh * 100] for p, sh in zip(periods, s["share"])],
            hovertemplate=(f"<b>{label[tid]}</b><br>%{{customdata[0]}} · rank %{{y:.0f}}"
                           "<br>%{customdata[1]:.1f}% of papers<extra></extra>"),
        ))

    fig.update_yaxes(
        autorange="reversed", tickmode="array", tickvals=list(range(1, n + 1)),
        ticktext=[str(i) for i in range(1, n + 1)],
        title_text="rank by share of papers  (1 = most-studied)",
        showgrid=True, gridcolor="rgba(120,130,150,0.10)", zeroline=False,
        range=[n + 0.5, 0.5],
    )
    fig.update_xaxes(
        tickmode="array", tickvals=centres, ticktext=[str(int(c)) for c in centres],
        range=[centres[0] - 1.5, centres[-1] + 1.5],
        showgrid=False, ticks="outside", tickcolor="rgba(120,130,150,0.4)",
        tickfont=dict(family=FONT, size=11),
    )
    fig.update_layout(
        title=dict(
            text=("<b>The shifting pecking order of ICSE research</b>"
                  "<br><span style='font-size:12.5px;color:#9098a3'>"
                  "each theme's rank by share of papers · top = most-studied · "
                  f"a line crossing = one theme overtaking another · {BUCKET}-year periods</span>"),
            x=0.012, xanchor="left", y=0.96, yanchor="top",
            font=dict(family=FONT, size=19, color="#23272e"),
        ),
        legend=dict(title=dict(text="Theme"), x=1.015, y=1.0, xanchor="left",
                    yanchor="top", font=dict(family=FONT, size=11.5),
                    bordercolor="#e2e4ea", borderwidth=1, bgcolor="rgba(255,255,255,0.8)"),
        template="plotly_white", height=640,
        font=dict(family=FONT, size=11, color="#3a3f4a"),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=104, l=64, r=250, b=52),
        hoverlabel=dict(align="left", bgcolor="white", bordercolor="#e2e4ea",
                        font=dict(family=FONT, size=12, color="#2a2e36")),
    )
    save(fig, NAME, post_script=_POST_SCRIPT)


def main():
    centres, periods, series, label, color = build()
    moves = {label[t]: s["rank"][0] - s["rank"][-1] for t, s in series.items()}
    riser = max(moves, key=moves.get)
    print(f"Theme rank trends: {len(series)} themes over {len(periods)} {BUCKET}-year periods "
          f"({periods[0]} … {periods[-1]}) · biggest climber: {riser} (+{int(moves[riser])} ranks)")
    plot(centres, periods, series, label, color)


if __name__ == "__main__":
    main()
