"""
researcher_migration_sankey.py — A scrolling, zooming timeline of how ICSE
researchers migrate between topics, one 5-year window at a time.

WHAT IT DOES
  An alluvial migration diagram drawn on a real time axis. For every researcher
  we find their *dominant* topic in each 5-year window (the topic they published
  in most), then track where they go in the next window. Topics are horizontal
  lanes (ordered by median year: older themes on top, newer below); each window
  is a column; and a ribbon flows from a researcher's dominant topic in one
  window to the next, its width = the number of researchers making that move.

  The whole 1995 -> 2025 timeline is drawn once, but the camera starts *zoomed
  in* on the first transition and then scrolls smoothly left to right across the
  field's history — 1995-1999 -> 2000-2004 -> ... -> 2020-2024 -> 2025 — before
  pulling back in the final frame to reveal the entire timeline at once. Because
  the motion is just the x-axis range gliding (and widening), the playback is a
  continuous scroll-then-zoom, not a series of resets.

      grey, horizontal ribbons  = researchers who stayed in their topic
      coloured ribbons          = researchers who migrated (toward newer themes,
                                  so downward)
      node size                 = researchers passing through that topic/window

WHAT IT HOPES TO ANSWER
  Topic drift is usually told as papers appearing and disappearing — but papers
  are written by people, so *where do the people go*? Scrolling the timeline
  shows which eras churned (lots of migration) versus stayed loyal, which topics
  recruited talent and which fed it elsewhere, and how the pace of movement
  accelerates as the field grows.

  Restricted to the TOP_K largest topics and flows of >= MIN_AUTHORS researchers
  so each view shows the main currents, not every eddy.

Reads:  data/processed/icse_paper_topics.parquet, data/processed/icse_topics.parquet,
        data/interim/icse_enriched.parquet  (all via _common)
Writes: outputs/figures/researcher_migration_sankey.html
"""

import json
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from topicdrift.constants import OUTLIER_TOPIC_ID
from topicdrift.visualization._common import (
    PROCESSED_DIR,
    clean_author,
    load_paper_topics,
    load_topics,
    save,
)

log = logging.getLogger(__name__)

NAME = "researcher_migration_sankey"
TOP_K = 10  # the 10 themes (all of them; kept as a guard if more are ever added)
BUCKET = 5  # window width in years
START = 1975  # start of the data (1976 falls in the 1975-1979 bucket)
MIN_AUTHORS = 2  # a flow needs at least this many researchers to be drawn
ZOOM_PAD = 0.28  # how much margin around the two columns the zoomed camera shows


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _win_label(b: int) -> str:
    return f"{b}-{b + BUCKET - 1}"


def build():
    """Return per-window flows, the window list, median-year-ordered top topics
    and labels.

    flows : {bucket: {(src_topic, tgt_topic): researcher_count}} for each window
            that has a successor; src is the dominant topic in `bucket`, tgt the
            dominant topic in `bucket + BUCKET`.
    """
    pt = load_paper_topics()
    topics_tbl = load_topics()
    labels = dict(zip(topics_tbl["topic_id"].astype(int), topics_tbl["label"].astype(str)))

    top_ids = [
        int(t)
        for t in topics_tbl[topics_tbl["topic_id"] != OUTLIER_TOPIC_ID]
        .sort_values("size", ascending=False)
        .head(TOP_K)["topic_id"]
    ]
    top_set = set(top_ids)

    pt = pt[pt["topic_id"].isin(top_set)].copy()
    pt["bucket"] = (pt["year"] // BUCKET * BUCKET).astype(int)
    pt = pt[pt["bucket"] >= START]

    # older themes on top, newer below -> migration reads as downward drift
    median_year = pt.groupby("topic_id")["year"].median()
    top_ids = sorted(top_ids, key=lambda t: median_year.get(t, 0))

    by_author: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for row in pt.itertuples():
        if row.authors is None:
            continue
        for a in list(row.authors):
            by_author[clean_author(a)][row.bucket][int(row.topic_id)] += 1

    flows: dict[int, Counter] = defaultdict(Counter)
    for _author, per_window in by_author.items():
        dominant = {
            b: sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            for b, counter in per_window.items()
        }
        for b in sorted(dominant):
            if b + BUCKET in dominant:
                flows[b][(dominant[b], dominant[b + BUCKET])] += 1

    buckets = sorted(b for b in pt["bucket"].unique())
    flows = {b: {k: v for k, v in c.items() if v >= MIN_AUTHORS} for b, c in flows.items()}
    flows = {b: c for b, c in flows.items() if c}
    return flows, buckets, top_ids, labels


def _ribbon(x0, y0, x1, y1, nseg=10):
    """Smooth S-curve points from (x0, y0) to (x1, y1) for a flow ribbon."""
    t = np.linspace(0, 1, nseg)
    s = t * t * (3 - 2 * t)  # smoothstep easing on the vertical move
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * s


# JS injected after the plot: a clickable row of period chips under the timeline
# and a panel that fills with that period's drift summary. {plot_id} is replaced
# by Plotly; SUMMARIES/ORDER/CHIP are substituted below (braces only in the CSS,
# which Plotly's literal {plot_id} replace leaves alone).
_POST_SCRIPT = """
var SUMMARIES = __SUMMARIES__, ORDER = __ORDER__, CHIP = __CHIP__;
var gd = document.getElementById("{plot_id}");

var style = document.createElement("style");
style.textContent =
  "#drift-periods{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-family:Inter,sans-serif;padding:8px 14px 2px;max-width:1100px;margin:0 auto}"
+ "#drift-periods .lbl{color:#8a8f98;font-size:12px;margin-right:4px}"
+ "#drift-periods button{background:#f3f4f7;border:1px solid #d7dae0;color:#3a3f4a;font-size:12px;padding:4px 9px;border-radius:6px;cursor:pointer;transition:background .12s}"
+ "#drift-periods button:hover{background:#e9ebef}"
+ "#drift-periods button.active{background:#2a3f5f;border-color:#2a3f5f;color:#fff;font-weight:600}"
+ "#drift-summary{font-family:Inter,sans-serif;max-width:1100px;margin:6px auto 22px;padding:0 16px;color:#222}"
+ "#drift-summary h3{color:#23272e;margin:.2em 0}#drift-summary ul{padding-left:20px;margin:.2em 0}#drift-summary li{margin:2px 0;color:#444}"
+ "body{transition:background .2s,color .2s}"
+ "body.dark{background:#1e1e1e;color:#e6e6e6}"
+ "body.dark #drift-periods .lbl{color:#9aa0a8}"
+ "body.dark #drift-periods button{background:#2a2a2a;border-color:#555;color:#e0e0e0}"
+ "body.dark #drift-periods button:hover{background:#3a3a3a}"
+ "body.dark #drift-periods button.active{background:#4a90d9;border-color:#4a90d9;color:#fff}"
+ "body.dark #drift-summary{color:#cfd3d9}"
+ "body.dark #drift-summary h3{color:#f0f0f0!important}"
+ "body.dark #drift-summary p,body.dark #drift-summary li,body.dark #drift-summary b{color:#c4c8d0!important}"
+ "#td-dm-wrap{color:#3a3f4a}body.dark #td-dm-wrap{color:#cfd3d9}";
document.head.appendChild(style);

var bar = document.createElement("div");
bar.id = "drift-periods";
bar.innerHTML = "<span class='lbl'>Researcher drift in a period:</span>";
ORDER.forEach(function(b){
  var btn = document.createElement("button");
  btn.textContent = CHIP[b];
  btn.onclick = function(){
    document.getElementById("drift-summary").innerHTML = SUMMARIES[b] || "";
    bar.querySelectorAll("button").forEach(function(x){ x.classList.remove("active"); });
    btn.classList.add("active");
  };
  bar.appendChild(btn);
});
gd.parentNode.insertBefore(bar, gd.nextSibling);

var panel = document.createElement("div");
panel.id = "drift-summary";
panel.innerHTML = "<p style='color:#777'>Click a period above to see who drifted where.</p>";
gd.parentNode.insertBefore(panel, bar.nextSibling);

// ── dark mode (toggle when standalone; obeys a parent page's td-dark message) ──
function applyDark(on){
  document.body.classList.toggle("dark", !!on);
  Plotly.relayout(gd, on ? {
    paper_bgcolor:"#1e1e1e", plot_bgcolor:"#1e1e1e", "font.color":"#e6e6e6",
    "title.font.color":"#f0f0f0",
    "updatemenus[0].bgcolor":"#2a2a2a", "updatemenus[0].bordercolor":"#555",
    "updatemenus[0].font.color":"#e6e6e6", "sliders[0].bgcolor":"#3a3a3a"
  } : {
    paper_bgcolor:"white", plot_bgcolor:"white", "font.color":"#3a3f4a",
    "title.font.color":"#23272e",
    "updatemenus[0].bgcolor":"#f3f4f7", "updatemenus[0].bordercolor":"#d7dae0",
    "updatemenus[0].font.color":"#3a3f4a", "sliders[0].bgcolor":"#dfe2e8"
  });
}
if (window.self === window.top){            // standalone: toggle pinned under Play/Pause
  var dm = document.createElement("div");
  dm.id = "td-dm-wrap";
  dm.style.cssText = "position:absolute;top:88px;right:38px;z-index:1000;font-family:Inter,sans-serif;font-size:12px;display:flex;align-items:center;gap:7px";
  dm.innerHTML = "<span>Dark mode</span><label style='position:relative;display:inline-block;width:40px;height:21px'>"
    + "<input type='checkbox' id='td-dm' style='opacity:0;width:0;height:0'>"
    + "<span id='td-sw' style='position:absolute;cursor:pointer;inset:0;background:#ccc;border-radius:22px;transition:.2s'>"
    + "<span id='td-knob' style='position:absolute;height:15px;width:15px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.2s'></span></span></label>";
  gd.style.position = "relative";
  gd.appendChild(dm);
  document.getElementById("td-dm").addEventListener("change", function(e){
    var on = e.target.checked;
    applyDark(on);
    document.getElementById("td-sw").style.background = on ? "#4a90d9" : "#ccc";
    document.getElementById("td-knob").style.transform = on ? "translateX(19px)" : "";
  });
}
window.addEventListener("message", function(ev){
  if (ev.data && ev.data.type === "td-dark"){ applyDark(ev.data.on); }
});
"""


def _period_summaries(transitions, flows, labels, color_of):
    """Per-transition HTML drift summaries keyed by the start bucket (as a string),
    plus the chip order and labels for the clickable period bar."""
    def chip(tid):
        return (f"<span style='color:{color_of.get(tid, '#888888')};font-weight:600'>"
                f"{labels[tid]}</span>")

    summaries, order, chip_labels = {}, [], {}
    for b in transitions:
        fl = flows[b]
        total = sum(fl.values())
        stayed = sum(v for (s, t), v in fl.items() if s == t)
        migrated = total - stayed
        moves = sorted(((s, t, v) for (s, t), v in fl.items() if s != t),
                       key=lambda x: -x[2])[:6]

        inflow, outflow, from_t, stay_t = (defaultdict(int) for _ in range(4))
        for (s, t), v in fl.items():
            from_t[s] += v
            if s == t:
                stay_t[s] += v
            else:
                outflow[s] += v
                inflow[t] += v
        net = {th: inflow[th] - outflow[th] for th in set(inflow) | set(outflow)}
        gainers = [x for x in sorted(net.items(), key=lambda x: -x[1]) if x[1] > 0][:3]
        losers = [x for x in sorted(net.items(), key=lambda x: x[1]) if x[1] < 0][:3]
        loyal = {th: stay_t[th] / from_t[th] for th in from_t if from_t[th] >= 5}
        most_loyal = max(loyal.items(), key=lambda x: x[1]) if loyal else None

        def pct(n):
            return f"{round(100 * n / total)}%" if total else "0%"

        h = [f"<h3>{_win_label(b)} &#8594; {_win_label(b + BUCKET)}</h3>",
             f"<p style='color:#555;margin:.2em 0 .8em'><b>{total}</b> researchers tracked"
             f" · <b>{stayed}</b> stayed ({pct(stayed)})"
             f" · <b>{migrated}</b> moved ({pct(migrated)})</p>"]
        if moves:
            items = "".join(f"<li>{v} · {chip(s)} <span style='color:#aaa'>&#8594;</span> "
                            f"{chip(t)}</li>" for s, t, v in moves)
            h.append(f"<p style='margin:.3em 0;color:#444'><b>Biggest moves</b></p>"
                     f"<ul>{items}</ul>")
        if gainers:
            h.append("<p style='margin:.4em 0;color:#444'><b>Net gain:</b> "
                     + ", ".join(f"{chip(th)} +{n}" for th, n in gainers) + "</p>")
        if losers:
            h.append("<p style='margin:.2em 0;color:#444'><b>Net loss:</b> "
                     + ", ".join(f"{chip(th)} {n}" for th, n in losers) + "</p>")
        if most_loyal:
            h.append(f"<p style='margin:.4em 0;color:#444'><b>Most loyal:</b> "
                     f"{chip(most_loyal[0])} ({round(100 * most_loyal[1])}% stayed)</p>")

        key = str(b)
        summaries[key] = "".join(h)
        order.append(key)
        chip_labels[key] = _win_label(b)
    return summaries, order, chip_labels


def plot(flows, buckets, top_ids, labels):
    # canonical theme colours (shared with the bump chart / treemap) keyed by name
    ct = pd.read_parquet(PROCESSED_DIR / "conf_topics.parquet")
    cmap = dict(zip(ct["group"].astype(str), ct["group_color"].astype(str)))
    color_of = {tid: cmap.get(labels[tid], "#888888") for tid in top_ids}
    rank = {tid: i for i, tid in enumerate(top_ids)}
    n_topics = len(top_ids)

    transitions = [b for b in buckets if b in flows]
    timeline = sorted({b for b in transitions} | {b + BUCKET for b in transitions})
    col = {b: i for i, b in enumerate(timeline)}

    def yof(t):
        return n_topics - 1 - rank[t]  # oldest topic on top

    # node researcher counts = max of flow in / flow out (how many pass through)
    out_sum, in_sum, present = defaultdict(int), defaultdict(int), set()
    for b in transitions:
        for (s, t), n in flows[b].items():
            out_sum[(b, s)] += n
            in_sum[(b + BUCKET, t)] += n
            present.update({(b, s), (b + BUCKET, t)})
    node_val = {nd: max(out_sum.get(nd, 0), in_sum.get(nd, 0)) for nd in present}

    max_flow = max(n for b in transitions for n in flows[b].values())
    max_node = max(node_val.values())

    fig = go.Figure()

    # --- precompute each ribbon's full geometry once (drawn source -> target) ---
    STEPS = 88  # animation frames per column (higher = smoother/slower scroll)
    ribbons = []  # one entry per flow, in a fixed trace order
    for ti, b in enumerate(transitions):
        x0, x1 = col[b], col[b + BUCKET]
        for (s, t), n in sorted(flows[b].items(), key=lambda kv: -kv[1]):
            xs, ys = _ribbon(x0, yof(s), x1, yof(t))
            same = s == t
            ribbons.append(
                dict(
                    ti=ti,
                    xs=xs,
                    ys=ys,
                    width=1.5 + 7 * np.sqrt(n / max_flow),
                    color=("rgba(170,176,186,0.32)" if same else _hex_to_rgba(color_of[s], 0.62)),
                    text=(
                        f"<b>{n} researchers</b><br>{labels[s]} ({_win_label(b)})"
                        f"  &#8594;  {labels[t]} ({_win_label(b + BUCKET)})"
                    ),
                )
            )
    n_ribbons = len(ribbons)

    def _slice(r, frac):
        """(x, y) for a ribbon drawn to `frac` of its length — the snake so far.

        The drawn portion ends at an *exact interpolated* point at `frac` (not the
        nearest sample), so the growing tip advances continuously frame to frame
        rather than snapping point-to-point — that snapping is what looked choppy.
        """
        xs, ys = r["xs"], r["ys"]
        if frac <= 0:
            return xs[:0], ys[:0]
        if frac >= 1:
            return xs, ys
        pos = frac * (len(xs) - 1)
        k = int(pos)
        t = pos - k
        ex = xs[k] + (xs[k + 1] - xs[k]) * t
        ey = ys[k] + (ys[k + 1] - ys[k]) * t
        return np.append(xs[: k + 1], ex), np.append(ys[: k + 1], ey)

    # base traces: full styling, but start empty so they snake on during playback
    for r in ribbons:
        xs, ys = _slice(r, 0.0)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=r["width"], color=r["color"], shape="spline"),
                hoverinfo="text",
                text=r["text"],
                showlegend=False,
            )
        )

    # --- nodes ---
    nodes = list(present)
    fig.add_trace(
        go.Scatter(
            x=[col[b] for b, _t in nodes],
            y=[yof(t) for _b, t in nodes],
            mode="markers",
            marker=dict(
                size=[10 + 32 * np.sqrt(node_val[nd] / max_node) for nd in nodes],
                color=[color_of[t] for _b, t in nodes],
                line=dict(width=1.6, color="white"),
                opacity=0.96,
            ),
            text=[
                f"<b>{labels[t]}</b><br>{_win_label(b)}<br>{node_val[(b, t)]} researchers"
                for b, t in nodes
            ],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    last = len(timeline) - 1
    node_index = n_ribbons
    node_world_x = np.array([col[b] for b, _t in nodes], dtype=float)
    win_world_x = np.array([col[b] for b in timeline], dtype=float)
    win_text = [_win_label(b) for b in timeline]

    FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"

    HALF = 0.5 + ZOOM_PAD  # half-width of the zoomed view
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(win_world_x),
        ticktext=win_text,
        range=[-HALF, HALF],
        showgrid=False,
        ticks="",
        tickfont=dict(family=FONT, size=12, color="#8a8f98"),
        showline=False,
    )
    # y tick labels are replaced by colour-coded annotations (below); keep only the
    # faint lane gridlines here.
    fig.update_yaxes(
        tickmode="array",
        tickvals=[yof(t) for t in top_ids],
        showticklabels=False,
        ticks="",
        range=[-0.9, n_topics - 0.1],
        showgrid=True,
        gridcolor="rgba(120,130,150,0.10)",
        zeroline=False,
    )

    # one colour-coded lane label per topic, pinned just left of the plot so they
    # stay put while the ribbons scroll underneath
    lane_labels = [
        dict(
            x=-0.008,
            y=yof(t),
            xref="paper",
            yref="y",
            xanchor="right",
            yanchor="middle",
            text=labels[t],
            showarrow=False,
            font=dict(family=FONT, size=11, color=color_of[t]),
        )
        for t in top_ids
    ]

    # The axis range NEVER changes (animating it is what made the camera shake).
    # Instead we scroll the world underneath a fixed camera: each frame maps world
    # x -> screen x via (x - focus) * scale. During the scroll, scale = 1 and the
    # focus (= the draw-frontier) glides left to right, so each ribbon is drawn to
    # fraction clamp(focus - i, 0, 1) with its tip parked at screen 0 (centre). The
    # finale eases the scale down and the focus to the timeline centre to fit the
    # whole thing. Only trace data changes, so the motion is perfectly steady.
    by_ti = {ti: [i for i, r in enumerate(ribbons) if r["ti"] == ti] for ti in range(last)}

    def xticks(focus, scale):
        return dict(tickvals=list((win_world_x - focus) * scale), ticktext=win_text)

    def node_scatter(focus, scale):
        return go.Scatter(x=(node_world_x - focus) * scale)

    def ribbon_scatter(r, frac, focus, scale):
        x, y = _slice(r, frac)
        return go.Scatter(x=(x - focus) * scale, y=y)

    frames, slider_steps = [], []

    empty = go.Scatter(x=[], y=[])
    all_idx = list(range(n_ribbons)) + [node_index]
    play_names = []

    # --- PLAY frames (the snake): light & sequential. Each step only re-sends the
    # windows in view, with the active one drawn to clamp(focus - i, 0, 1) so it
    # grows. p0 clears everything for a clean replay; off-screen ribbons just stay
    # parked off-frame, so playback stays cheap and the lines draw on. ---
    for k in range(last * STEPS + 1):
        f = k / STEPS
        if k == 0:
            idx = list(range(n_ribbons))
            data = [empty for _ in ribbons]
        else:
            vis = [ti for ti in range(last) if ti <= f + HALF and ti + 1 >= f - HALF]
            idx = [i for ti in vis for i in by_ti[ti]]
            data = [
                ribbon_scatter(ribbons[i], min(max(f - ribbons[i]["ti"], 0.0), 1.0), f, 1.0)
                for i in idx
            ]
        data.append(node_scatter(f, 1.0))
        frames.append(
            go.Frame(
                name=f"p{k}",
                data=data,
                traces=idx + [node_index],
                layout=go.Layout(xaxis=xticks(f, 1.0)),
            )
        )
        play_names.append(f"p{k}")

    # --- SCRUB frames (for the slider): lines are FIXED. Every in-view ribbon is
    # drawn in full and the camera just pans; nothing grows or erases as you drag.
    # Each frame is self-complete (off-screen ribbons sent empty) so scrubbing
    # forwards, backwards or jumping always lands on a correct, fully-drawn scene. ---
    SCRUB = 12  # slider stops per column
    for m in range(last * SCRUB + 1):
        f = m / SCRUB
        data = [
            (
                ribbon_scatter(r, 1.0, f, 1.0)
                if (r["ti"] <= f + HALF and r["ti"] + 1 >= f - HALF)
                else empty
            )
            for r in ribbons
        ]
        data.append(node_scatter(f, 1.0))
        frames.append(
            go.Frame(
                name=f"s{m}",
                data=data,
                traces=all_idx,
                layout=go.Layout(xaxis=xticks(f, 1.0)),
            )
        )
        slider_steps.append(("", f"s{m}"))

    # --- zoom-out finale (shared by Play and the slider) ---
    ZOOM_OUT = 16
    scale_end = (2 * HALF - 0.22) / last
    for j in range(1, ZOOM_OUT + 1):
        a = j / ZOOM_OUT
        a = a * a * (3 - 2 * a)  # smoothstep ease
        focus = last + (last / 2 - last) * a
        scale = 1 + (scale_end - 1) * a
        data = [ribbon_scatter(r, 1.0, focus, scale) for r in ribbons]
        data.append(node_scatter(focus, scale))
        frames.append(
            go.Frame(
                name=f"zoom{j}",
                data=data,
                traces=all_idx,
                layout=go.Layout(xaxis=xticks(focus, scale)),
            )
        )
        play_names.append(f"zoom{j}")
        slider_steps.append(("", f"zoom{j}"))
    fig.frames = frames

    # Play animates the snake frames (then the zoom-out) explicitly, always from the
    # start; the slider scrubs the fixed-line frames. Fixed axis -> no range tween,
    # so redraw each tiny step with no transition: steady, shake-free, ~30 fps.
    play = {
        "frame": {"duration": 33, "redraw": True},
        "transition": {"duration": 0},
        "mode": "immediate",
    }
    pause = {
        "frame": {"duration": 0, "redraw": False},
        "mode": "immediate",
        "transition": {"duration": 0},
    }
    step_anim = {
        "frame": {"duration": 0, "redraw": True},
        "mode": "immediate",
        "transition": {"duration": 0},
    }

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=1.0,
                y=1.11,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=0),
                bgcolor="#f3f4f7",
                bordercolor="#d7dae0",
                borderwidth=1,
                font=dict(family=FONT, size=12, color="#3a3f4a"),
                buttons=[
                    dict(
                        label="&#9654;&nbsp; Play",
                        method="animate",
                        args=[play_names, play],
                    ),
                    dict(
                        label="&#10074;&#10074;&nbsp; Pause",
                        method="animate",
                        args=[[None], pause],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.12,
                len=0.84,
                pad=dict(t=16, b=4),
                currentvalue=dict(visible=False),  # many stops -> no per-step readout
                ticklen=0,
                minorticklen=0,
                borderwidth=0,
                bgcolor="#dfe2e8",
                bordercolor="rgba(0,0,0,0)",
                steps=[
                    dict(method="animate", label=label, args=[[frame], step_anim])
                    for label, frame in slider_steps
                ],
            )
        ],
        title=dict(
            text=(
                "<b>How ICSE researchers migrate between themes</b>"
                "<br><span style='font-size:12.5px;color:#9098a3'>"
                "drag to scroll, or press play · ribbon width = researchers · "
                "grey = stayed, colour = migrated · newer themes lower</span>"
            ),
            x=0.012,
            xanchor="left",
            y=0.96,
            yanchor="top",
            font=dict(family=FONT, size=19, color="#23272e"),
        ),
        annotations=lane_labels,
        template="plotly_white",
        height=760,
        font=dict(family=FONT, size=11, color="#3a3f4a"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=118, l=215, r=34, b=58),
        hoverlabel=dict(
            align="left",
            bgcolor="white",
            bordercolor="#e2e4ea",
            font=dict(family=FONT, size=12, color="#2a2e36"),
        ),
    )
    summaries, order, chip_labels = _period_summaries(transitions, flows, labels, color_of)
    post = (
        _POST_SCRIPT
        .replace("__SUMMARIES__", json.dumps(summaries, ensure_ascii=False).replace("</", "<\\/"))
        .replace("__ORDER__", json.dumps(order))
        .replace("__CHIP__", json.dumps(chip_labels, ensure_ascii=False))
    )
    save(fig, NAME, post_script=post)


def main():
    flows, buckets, top_ids, labels = build()
    moved = sum(v for c in flows.values() for (s, t), v in c.items() if s != t)
    stayed = sum(v for c in flows.values() for (s, t), v in c.items() if s == t)
    log.info(
        "Migration timeline: %d windows, %d flows (>= %d researchers) — %d stayed, %d migrated",
        len(flows),
        sum(len(c) for c in flows.values()),
        MIN_AUTHORS,
        stayed,
        moved,
    )
    plot(flows, buckets, top_ids, labels)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
