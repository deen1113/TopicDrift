"""
lexical_turnover.py — how much each of the 10 ICSE themes churns its own
defining vocabulary from one 5-year window to the next.

WHAT IT DOES
  A theme is a fixed cluster of papers, but the words that *define* it are
  recomputed for every 5-year window (c-TF-IDF over just that window's papers),
  so a theme's language drifts even when its identity doesn't. For each window we
  compare a theme's five defining words with the previous window's and measure the
  turnover = 1 - Jaccard: 0 = the same five words, 1 = a complete swap.

  The grid has one row per theme and one column per window, shaded by turnover.
  Hovering a cell reveals exactly which words entered (+) and dropped (−) — so the
  detail is on demand rather than crowding every cell. A bar strip below, on its
  own timeline, tracks the field-wide average churn window by window.

      cell shade   = how much of the five-word vocabulary turned over
      hover        = the words gained (+) and lost (−), and the current five
      bottom strip = average turnover across the 10 themes that window

WHAT IT HOPES TO ANSWER
  "Topic drift" usually means themes rising and falling; this asks a finer
  question — does a surviving theme keep talking about the same things? A pale row
  is a theme with settled language; a row that keeps flaring is one repeatedly
  absorbing new sub-paradigms.

Reads:  data/processed/icse_topics_over_time.parquet, data/processed/icse_topics.parquet
        (via _common)
Writes: outputs/figures/lexical_turnover.html
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from topicdrift.visualization._common import load_topics, load_tot, save

NAME = "lexical_turnover"
TOP_K = 10        # the 10 themes
BUCKET = 5        # window width in years
MIN_FREQ = 3      # a window needs >= this many papers to count (drop 1-paper noise)

# A warm peach → magenta → plum sequential, but with a sharper ramp through the
# mid-to-high range (where most turnover values sit) so cells differentiate
# crisply. Light low end (not white) and a deep-but-not-black top keep it readable
# on both light and dark backgrounds.
SCALE = [
    [0.00, "#fde4c0"],
    [0.22, "#f8a86c"],
    [0.42, "#f06b6b"],
    [0.60, "#e0436f"],
    [0.78, "#c0247a"],
    [1.00, "#5e1668"],
]

# Dark-mode toggle (top-right) that recolours the figure; also obeys a parent
# page's `td-dark` postMessage when embedded.
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
    paper_bgcolor:"#1e1e1e", plot_bgcolor:"#272430", "font.color":"#e6e6e6",
    "title.font.color":"#f0f0f0"
  } : {
    paper_bgcolor:"white", plot_bgcolor:"#f2eef5", "font.color":"#3a3f4a",
    "title.font.color":"#23272e"
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


def _word_list(s) -> list[str]:
    """Defining words for a window, in c-TF-IDF rank order (deduplicated)."""
    out, seen = [], set()
    for w in str(s).split(","):
        w = w.strip()
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _win(b: int) -> str:
    return f"{b}–{str(b + 4)[2:]}"


def build():
    """Turnover matrix, per-cell hover detail (scenario-aware), the per-window
    field mean, and the axis labels."""
    tot = load_tot()
    t = load_topics()
    labels = dict(zip(t["topic_id"].astype(int), t["label"].astype(str)))
    sizes = t.set_index("topic_id")["size"]

    top_ids = [i for i in sizes.sort_values(ascending=False).index
               if i in labels][:TOP_K]

    tot = tot[tot["topic_id"].isin(top_ids) & (tot["freq"] >= MIN_FREQ)].copy()
    transitions = sorted(tot["year_bucket"].unique())[1:]  # column = window vs previous

    # the defining words for each theme in every window it is active
    present: dict[int, dict[int, list[str]]] = {tid: {} for tid in top_ids}
    for _, r in tot.iterrows():
        present[int(r["topic_id"])][int(r["year_bucket"])] = _word_list(r["top_words"])

    z = np.full((len(top_ids), len(transitions)), np.nan)
    hov = np.full((len(top_ids), len(transitions)), "", dtype=object)
    grey, gain, loss = "#9098a3", "#2fae7a", "#e0726a"

    for ri, tid in enumerate(top_ids):
        pres = present[tid]
        head = f"<b>{labels[tid]}</b>&nbsp;&nbsp;"
        for ci, b in enumerate(transitions):
            if b not in pres:  # theme had no papers in this window
                hov[ri, ci] = f"{head}{_win(b)}<br><span style='color:{grey}'>No papers available</span>"
                continue
            cur = pres[b]
            earlier = [pb for pb in pres if pb < b]
            if not earlier:  # theme's first active window — nothing to compare to
                hov[ri, ci] = (f"{head}{_win(b)}<br><span style='color:{grey}'>"
                               f"First active window — no prior to compare</span>"
                               f"<br><span style='color:{grey}'>terms: {', '.join(cur)}</span>")
                continue
            prev = pres[max(earlier)]
            prev_set, cur_set = set(prev), set(cur)
            turn = 1 - len(prev_set & cur_set) / len(prev_set | cur_set)
            z[ri, ci] = turn
            span = f"{_win(max(earlier))} → {_win(b)}"
            if turn == 0:  # identical five words
                hov[ri, ci] = (f"{head}{span}<br>No change in vocabulary"
                               f"<br><span style='color:{grey}'>terms: {', '.join(cur)}</span>")
            else:
                entered = [w for w in cur if w not in prev_set]
                left = [w for w in prev if w not in cur_set]
                hov[ri, ci] = (
                    f"{head}{span}<br>turnover {turn:.0%}"
                    f"<br><span style='color:{gain}'>entered: {', '.join(entered) or '—'}</span>"
                    f"<br><span style='color:{loss}'>dropped: {', '.join(left) or '—'}</span>"
                    f"<br><span style='color:{grey}'>now: {', '.join(cur)}</span>"
                )

    field_mean = np.nanmean(z, axis=0)
    x_labels = [_win(b) for b in transitions]
    y_labels = [labels[i] for i in top_ids]
    return z, hov, field_mean, x_labels, y_labels


def plot(z, hov, field_mean, x_labels, y_labels):
    FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, row_heights=[0.83, 0.17],
        vertical_spacing=0.13,
    )

    # main grid — turnover shade only (the words live in the hover, not the cells)
    fig.add_trace(go.Heatmap(
        z=z, x=x_labels, y=y_labels, customdata=hov,
        colorscale=SCALE, zmin=0, zmax=1, hoverongaps=True,
        hovertemplate="%{customdata}<extra></extra>", xgap=3, ygap=3,
        colorbar=dict(title=dict(text="Vocabulary<br>turnover", side="top"),
                      thickness=13, len=0.62, y=0.78, yanchor="middle",
                      tickvals=[0, 0.5, 1], ticktext=["steady", "half", "renewed"],
                      tickfont=dict(size=10), outlinewidth=0),
    ), row=1, col=1)

    # bottom strip — field-wide average churn, on its own timeline
    fig.add_trace(go.Bar(
        x=x_labels, y=field_mean,
        marker=dict(color=field_mean, colorscale=SCALE, cmin=0, cmax=1, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>average turnover of the 10 themes: %{y:.0%}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig.update_yaxes(autorange="reversed", tickfont=dict(family=FONT, size=11),
                     showgrid=False, row=1, col=1)
    fig.update_xaxes(tickfont=dict(family=FONT, size=11), showgrid=False,
                     side="bottom", row=1, col=1)

    fig.update_yaxes(title_text="Mean turnover", range=[0, float(np.nanmax(field_mean)) * 1.05],
                     tickvals=[0, 0.5], tickformat=".0%",
                     tickfont=dict(family=FONT, size=9), showgrid=False, row=2, col=1)
    fig.update_xaxes(title_text="Five-year window  (vocabulary compared with the preceding window)",
                     tickfont=dict(family=FONT, size=11), showgrid=False, row=2, col=1)

    fig.update_layout(
        title=dict(
            text=("<b>Vocabulary turnover within ICSE research themes</b>"
                  "<br><span style='font-size:12.5px;color:#9098a3'>"
                  "proportion of each theme's five defining words replaced per five-year "
                  "window · darker = greater turnover · hover for the terms gained (+) and "
                  "lost (−) · grey = theme not yet active</span>"),
            x=0.012, xanchor="left", y=0.965, yanchor="top",
            font=dict(family=FONT, size=19, color="#23272e"),
        ),
        template="plotly_white", height=740,
        font=dict(family=FONT, size=11, color="#3a3f4a"),
        paper_bgcolor="white", plot_bgcolor="#f2eef5",
        margin=dict(t=120, l=205, r=104, b=58),
        bargap=0.2,
        hoverlabel=dict(align="left", bgcolor="white", bordercolor="#e2e4ea",
                        font=dict(family=FONT, size=12, color="#2a2e36")),
    )
    save(fig, NAME, post_script=_POST_SCRIPT)


def main():
    z, hov, field_mean, x_labels, y_labels = build()
    busiest = x_labels[int(np.nanargmax(field_mean))]
    print(f"Lexical turnover: {len(y_labels)} themes × {len(x_labels)} windows · "
          f"avg churn peaks in {busiest} ({np.nanmax(field_mean):.0%})")
    plot(z, hov, field_mean, x_labels, y_labels)


if __name__ == "__main__":
    main()
