"""
topic_drift_search.py — Browse-by-theme drift explorer for ICSE.

WHAT IT DOES
  A two-level browser over the ICSE corpus on the global multi-conference fit:
  the ten overarching themes ("big 10") and, under each, all of its BERTopic
  sub-topics. Click a theme to overlay its share-over-time curve; expand it and
  click any sub-topic to overlay that sub-topic's curve too. Multiple series
  (themes and/or sub-topics) can be compared on the same axes, each with a star
  at its peak 5-year bucket. A Share % / Paper Count toggle switches the y-axis.
  Selecting a series also expands a card below the plot — sub-topic cards list
  the topic's top-cited papers as links; theme cards list their sub-topics.

WHAT IT HOPES TO ANSWER
  "How did this research theme — and the specific topics inside it — rise and
  fall at ICSE?" Complements the streamgraph (theme stack) and treemap
  (decade-grouped) by letting you pick any theme/sub-topic and compare drift.

Reads:  data/processed/conf_paper_topics.parquet, conf_topics.parquet
        data/interim/conf_enriched.parquet (titles), dblp_conf.parquet
        (authors + EE links), icse_enriched.parquet (citations, optional)
Writes: outputs/figures/topic_drift_search.html
"""

import json
import logging

import pandas as pd
import plotly.graph_objects as go

from topicdrift.constants import OUTLIER_TOPIC_ID
from topicdrift.visualization._common import (
    BUCKET_YEARS,
    FIGURES_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    conf_group_registry,
    load_conf_paper_topics,
    scope_filter,
)
# Reuse the treemap's per-paper link/author resolver (same package).
from topicdrift.visualization._treemap_data import _paper_links

log = logging.getLogger(__name__)

NAME = "topic_drift_search"
SCOPE = "icse"          # this explorer is ICSE-only (where TopicDrift began)
TOP_PAPERS = 3          # most-cited papers listed per sub-topic card


def _bucket_label(start: int, max_year: int) -> str:
    """Honest 5-year-bucket label: '1975–79', or just '2025' for a partial bucket."""
    end = min(start + BUCKET_YEARS - 1, max_year)
    return str(start) if end == start else f"{start}–{str(end)[2:]}"


def _enrich(keys: set[str]) -> pd.DataFrame:
    """Per-paper title + authors + EE link (+ citations where known) by dblp_key.

    Titles come from the pooled silver corpus; authors and the external link
    from the DBLP slice (via the shared resolver); citations from the ICSE
    silver table when present (the global conf corpus has none)."""
    out = _paper_links(keys)  # dblp_key, ee, authors_str
    titles = pd.read_parquet(INTERIM_DIR / "conf_enriched.parquet", columns=["dblp_key", "title"])
    out = out.merge(titles, on="dblp_key", how="left")
    icse = INTERIM_DIR / "icse_enriched.parquet"
    if icse.exists():
        cit = pd.read_parquet(icse, columns=["dblp_key", "citation_count"])
        out = out.merge(cit, on="dblp_key", how="left")
    else:
        out["citation_count"] = 0.0
    out["citation_count"] = out["citation_count"].fillna(0.0)
    return out


def _series(counts: pd.Series, buckets: list[int], bucket_total: pd.Series):
    """(freqs, shares, peak_bucket, peak_share) for a per-bucket count series."""
    freqs = [int(counts.get(b, 0)) for b in buckets]
    shares = [freqs[i] / int(bucket_total.get(b, 1)) for i, b in enumerate(buckets)]
    peak = max(range(len(shares)), key=lambda i: shares[i]) if shares else 0
    return freqs, shares, buckets[peak] if buckets else 0, (shares[peak] if shares else 0.0)


def build():
    """Return (buckets, labels, themes, subs) for the embedded JSON."""
    pt = scope_filter(load_conf_paper_topics(), SCOPE)
    pt = pt[pt["topic_id"] != OUTLIER_TOPIC_ID].dropna(subset=["group"]).copy()
    pt["topic_id"] = pt["topic_id"].astype(int)
    pt["bucket"] = (pt["year"] // BUCKET_YEARS) * BUCKET_YEARS

    buckets = sorted(int(b) for b in pt["bucket"].unique())
    max_year = int(pt["year"].max())
    labels = [_bucket_label(b, max_year) for b in buckets]
    bucket_total = pt.groupby("bucket").size()

    topics = pd.read_parquet(PROCESSED_DIR / "conf_topics.parquet")
    topics = topics[topics["topic_id"] != OUTLIER_TOPIC_ID]
    label_of = dict(zip(topics["topic_id"].astype(int), topics["llm_label"].astype(str)))
    words_of = {
        int(r.topic_id): [str(w) for w in r.top_words][:10] for r in topics.itertuples()
    }
    group_color, group_order = conf_group_registry()

    enr = _enrich(set(pt["dblp_key"]))
    paper_lut = pt[["dblp_key", "topic_id", "year"]].merge(enr, on="dblp_key", how="left")

    # ── sub-topics ───────────────────────────────────────────────────────────
    sub_counts = pt.groupby(["topic_id", "bucket"]).size()
    sub_total = pt.groupby("topic_id").size()
    theme_of = (
        pt.drop_duplicates("topic_id").set_index("topic_id")["group"].astype(str).to_dict()
    )

    subs: dict[str, dict] = {}
    for tid in sorted(pt["topic_id"].unique()):
        counts = sub_counts.loc[tid] if tid in sub_counts.index.get_level_values(0) else pd.Series(dtype=int)
        freqs, shares, peak_b, peak_s = _series(counts, buckets, bucket_total)
        theme = theme_of.get(tid, "Other")
        rows = (
            paper_lut[paper_lut["topic_id"] == tid]
            .sort_values("citation_count", ascending=False)
            .head(TOP_PAPERS)
        )
        papers = [
            {
                "t": (r.title if isinstance(r.title, str) and r.title else "(untitled)"),
                "y": int(r.year),
                "a": r.authors_str if isinstance(r.authors_str, str) else "",
                "u": r.ee if isinstance(r.ee, str) else "",
                "c": int(r.citation_count),
            }
            for r in rows.itertuples()
        ]
        subs[str(tid)] = {
            "key": f"t:{tid}",
            "label": label_of.get(tid, f"Topic {tid}"),
            "theme": theme,
            "color": group_color.get(theme, "#94a3b8"),
            "papers": int(sub_total.get(tid, 0)),
            "top_words": words_of.get(tid, []),
            "shares": shares,
            "freqs": freqs,
            "peak_bucket": peak_b,
            "peak_share": peak_s,
            "papers_list": papers,
        }

    # ── themes (the "big 10") ─────────────────────────────────────────────────
    theme_counts = pt.groupby(["group", "bucket"]).size()
    theme_total = pt.groupby("group").size()
    present = [g for g in group_order if g in set(pt["group"])]

    themes = []
    for g in present:
        counts = theme_counts.loc[g] if g in theme_counts.index.get_level_values(0) else pd.Series(dtype=int)
        freqs, shares, peak_b, peak_s = _series(counts, buckets, bucket_total)
        members = sorted(
            (tid for tid, t in subs.items() if t["theme"] == g),
            key=lambda tid: subs[tid]["papers"],
            reverse=True,
        )
        themes.append(
            {
                "key": f"g:{g}",
                "name": g,
                "color": group_color.get(g, "#94a3b8"),
                "papers": int(theme_total.get(g, 0)),
                "shares": shares,
                "freqs": freqs,
                "peak_bucket": peak_b,
                "peak_share": peak_s,
                "subs": members,
            }
        )

    return buckets, labels, themes, subs


_POST_SCRIPT = """
var BUCKETS = __BUCKETS__;
var LABELS  = __LABELS__;
var THEMES  = __THEMES__;   // [{key,name,color,papers,shares,freqs,peak_bucket,peak_share,subs:[tid,...]}]
var SUBS    = __SUBS__;     // {tid: {key,label,theme,color,papers,top_words,shares,freqs,peak_bucket,peak_share,papers_list}}

var gd = document.getElementById("{plot_id}");
var selected = new Set();   // keys: "g:<theme>" or "t:<id>"
var expanded = new Set();   // theme names currently expanded in the browser
var darkOn = false;

function byKey(k){ return k.charAt(0) === "g" ? themeByKey(k) : SUBS[k.slice(2)]; }
function themeByKey(k){ for (var i=0;i<THEMES.length;i++){ if (THEMES[i].key===k) return THEMES[i]; } return null; }

var style = document.createElement("style");
style.textContent =
  "body{font-family:sans-serif;margin:0;background:#fafbfc;color:#222;transition:background .2s,color .2s}"
+ "#search-wrap{padding:14px 18px;border-bottom:1px solid #e6e9ec;background:#fff}"
+ "#search-wrap h1{margin:0 0 4px;font-size:18px}"
+ "#search-wrap p{margin:0 0 10px;color:#666;font-size:13px}"
+ "#search-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}"
+ "#tools{display:flex;gap:6px}"
+ "#tools button{background:#fff;border:1px solid #cdd3da;color:#555;padding:6px 11px;font-size:12px;border-radius:13px;cursor:pointer}"
+ "#tools button:hover{background:#eef3f8}"
+ "#clear{border-color:#d8534f!important;color:#d8534f!important}"
+ "#clear:hover{background:#fdecec!important}"
+ "#main{display:grid;grid-template-columns:320px 1fr;gap:0;align-items:start}"
+ "#hits{max-height:560px;overflow-y:auto;border-right:1px solid #e6e9ec;background:#fff}"
+ ".thead{display:flex;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid #f0f2f4;cursor:pointer;font-size:13px}"
+ ".thead:hover{background:#f5f9fd}"
+ ".thead.on{background:#e7f0fa}"
+ ".caret{width:14px;color:#888;font-size:10px;flex:0 0 auto;text-align:center;cursor:pointer;user-select:none}"
+ ".sw{width:11px;height:11px;border-radius:3px;flex:0 0 auto}"
+ ".tname{font-weight:600;color:#222;flex:1}"
+ ".tcount{color:#999;font-size:11px}"
+ ".sub{display:flex;align-items:center;gap:8px;padding:6px 12px 6px 34px;border-bottom:1px solid #f5f6f8;cursor:pointer;font-size:12.5px}"
+ ".sub:hover{background:#f5f9fd}"
+ ".sub.on{background:#e7f0fa;border-left:3px solid #2a3f5f;padding-left:31px}"
+ ".sub .slbl{color:#333;flex:1}"
+ ".sub .scount{color:#aaa;font-size:11px}"
+ "#plot{min-height:520px}"
+ "#cards{padding:14px 18px;background:#fff;border-top:1px solid #e6e9ec}"
+ ".card{margin-bottom:14px;padding:11px 14px;border:1px solid #e6e9ec;border-radius:8px;background:#fafbfc}"
+ ".card h3{margin:0 0 4px;font-size:14px;display:flex;align-items:center;gap:7px}"
+ ".card .kw{color:#888;font-size:12px;font-style:italic;margin-bottom:6px}"
+ ".card ol{margin:6px 0 0;padding-left:22px;font-size:13px}"
+ ".card a{color:#2a3f5f;text-decoration:none}"
+ ".card a:hover{text-decoration:underline}"
+ ".card .meta{color:#999;font-size:12px}"
+ ".card .members{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}"
+ ".card .member{background:#eef3f8;border:1px solid #cdd9e5;color:#2a3f5f;padding:3px 9px;font-size:11.5px;border-radius:12px;cursor:pointer}"
+ ".card .member:hover{background:#dde6f0}"
+ ".card .member.on{background:#2a3f5f;color:#fff}"
+ "body.dark{background:#1e1e1e;color:#e6e6e6}"
+ "body.dark #search-wrap{background:#252525;border-bottom-color:#3a3a3a}"
+ "body.dark #search-wrap p{color:#aaa}"
+ "body.dark #tools button{background:#2a2a2a;border-color:#555;color:#ccc}"
+ "body.dark #tools button:hover{background:#34527a}"
+ "body.dark #hits{background:#252525;border-right-color:#3a3a3a}"
+ "body.dark .thead{border-bottom-color:#303030}"
+ "body.dark .thead:hover{background:#2a2a2a}"
+ "body.dark .thead.on{background:#2a3f5f}"
+ "body.dark .tname{color:#e6e6e6}"
+ "body.dark .sub{border-bottom-color:#2c2c2c}"
+ "body.dark .sub:hover{background:#2a2a2a}"
+ "body.dark .sub.on{background:#2a3f5f;border-left-color:#4a90d9}"
+ "body.dark .sub .slbl{color:#dcdcdc}"
+ "body.dark #cards{background:#252525;border-top-color:#3a3a3a}"
+ "body.dark .card{background:#2a2a2a;border-color:#3a3a3a}"
+ "body.dark .card .kw,body.dark .card .meta{color:#888}"
+ "body.dark .card a{color:#6db3f2}"
+ "body.dark .card .member{background:#2a3f5f;border-color:#3f5577;color:#cfe1f4}";
document.head.appendChild(style);

function applyDark(on){
  darkOn = !!on;
  document.body.classList.toggle("dark", darkOn);
  Plotly.relayout(gd, darkOn
    ? {paper_bgcolor:"#1e1e1e", plot_bgcolor:"#1e1e1e", "font.color":"#e6e6e6",
       "xaxis.gridcolor":"#3a3a3a", "yaxis.gridcolor":"#3a3a3a"}
    : {paper_bgcolor:"white", plot_bgcolor:"white", "font.color":"#2a3f5f",
       "xaxis.gridcolor":"#eee", "yaxis.gridcolor":"#eee"});
}
window.addEventListener("message", function(ev){
  if (ev.data && ev.data.type === "td-dark"){ applyDark(ev.data.on); }
});

var wrap = document.createElement("div");
wrap.id = "search-wrap";
wrap.innerHTML =
    "<h1>Topic Drift Search</h1>"
  + "<p>Browse the ten overarching themes and their sub-topics. Click a theme to plot its "
  + "drift; expand it (▸) to pick individual sub-topics. The y-axis is each series' "
  + "<b>share of ICSE papers</b> per period — hover any point for the raw paper count.</p>"
  + "<div id='search-row'>"
  +   "<div id='tools'>"
  +     "<button id='expand-all'>Expand all</button>"
  +     "<button id='collapse-all'>Collapse all</button>"
  +     "<button id='clear'>Clear selection</button>"
  +   "</div>"
  + "</div>";
gd.parentNode.insertBefore(wrap, gd);

var main = document.createElement("div"); main.id = "main";
var hits = document.createElement("div"); hits.id = "hits";
gd.parentNode.insertBefore(main, gd);
main.appendChild(hits);
main.appendChild(gd); gd.id = "plot";

var cards = document.createElement("div"); cards.id = "cards";
cards.innerHTML = "<p style='color:#999;font-size:13px'>Select a theme or sub-topic to see its details here.</p>";
main.parentNode.appendChild(cards);

function esc(s){ return (s+"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function renderBrowser(){
  var html = "";
  THEMES.forEach(function(g){
    var on = selected.has(g.key) ? " on" : "";
    var open = expanded.has(g.name);
    html += "<div class='thead" + on + "' data-key='" + esc(g.key) + "' data-theme='" + esc(g.name) + "'>"
         +    "<span class='caret' data-theme='" + esc(g.name) + "'>" + (open ? "\\u25be" : "\\u25b8") + "</span>"
         +    "<span class='sw' style='background:" + g.color + "'></span>"
         +    "<span class='tname'>" + esc(g.name) + "</span>"
         +    "<span class='tcount'>" + g.papers + " papers \\u00b7 " + g.subs.length + " topics</span>"
         +  "</div>";
    if (open){
      g.subs.forEach(function(tid){
        var s = SUBS[tid];
        var son = selected.has(s.key) ? " on" : "";
        html += "<div class='sub" + son + "' data-key='" + esc(s.key) + "'>"
             +    "<span class='sw' style='background:" + s.color + "'></span>"
             +    "<span class='slbl'>" + esc(s.label) + "</span>"
             +    "<span class='scount'>" + s.papers + "</span>"
             +  "</div>";
      });
    }
  });
  hits.innerHTML = html;
  hits.querySelectorAll(".caret").forEach(function(el){
    el.addEventListener("click", function(ev){ ev.stopPropagation(); toggleExpand(el.dataset.theme); });
  });
  hits.querySelectorAll(".thead").forEach(function(el){
    el.addEventListener("click", function(){ toggleSelect(el.dataset.key); });
  });
  hits.querySelectorAll(".sub").forEach(function(el){
    el.addEventListener("click", function(){ toggleSelect(el.dataset.key); });
  });
}

function toggleExpand(theme){
  if (expanded.has(theme)) expanded.delete(theme); else expanded.add(theme);
  renderBrowser();
}
function toggleSelect(key){
  if (selected.has(key)) selected.delete(key); else selected.add(key);
  renderBrowser(); redraw();
}

function seriesColor(item, i){
  // Themes keep their identity colour; sub-topics cycle through a palette so
  // several from the same theme stay distinguishable.
  if (item.key.charAt(0) === "g") return item.color;
  var PALETTE = ["#2563eb","#d97706","#16a34a","#9333ea","#dc2626","#0891b2","#ca8a04","#db2777","#4f46e5","#65a30d"];
  return PALETTE[i % PALETTE.length];
}

function redraw(){
  var traces = [], i = 0;
  selected.forEach(function(key){
    var t = byKey(key); if (!t) return;
    var nm = t.name || t.label;
    var col = seriesColor(t, i++);
    traces.push({
      x: LABELS, y: t.shares, customdata: t.freqs,
      mode: "lines+markers", name: nm, type: "scatter",
      line: {width: 2.5, color: col}, marker: {size: 6, color: col},
      hovertemplate: "<b>" + esc(nm) + "</b><br>%{x}: %{y:.1%} of ICSE papers"
                   + "<br>%{customdata} papers<extra></extra>"
    });
    var peakIdx = BUCKETS.indexOf(t.peak_bucket);
    if (peakIdx !== -1){
      traces.push({
        x: [LABELS[peakIdx]], y: [t.shares[peakIdx]],
        mode: "markers+text", type: "scatter",
        marker: {size: 14, symbol: "star", color: "rgba(0,0,0,0)",
                 line: {color: darkOn ? "#e6e6e6" : "#222", width: 1.5}},
        text: ["peak"], textposition: "top center",
        textfont: {size: 10, color: darkOn ? "#bbb" : "#555"},
        showlegend: false, hoverinfo: "skip"
      });
    }
  });
  var grid = darkOn ? "#3a3a3a" : "#eee";
  var layout = {
    title: selected.size ? "Drift curves — " + selected.size + " selected" : "Pick a theme or sub-topic to plot its drift",
    xaxis: {title: "5-year period", gridcolor: grid},
    yaxis: {title: "Share of ICSE papers", tickformat: ".0%", rangemode: "tozero", gridcolor: grid},
    legend: {orientation: "h", y: -0.18, x: 0},
    margin: {t: 50, l: 60, r: 20, b: 70}, template: "plotly_white",
    hovermode: "x unified",
    paper_bgcolor: darkOn ? "#1e1e1e" : "white",
    plot_bgcolor:  darkOn ? "#1e1e1e" : "white",
    font: {color: darkOn ? "#e6e6e6" : "#2a3f5f"}
  };
  Plotly.react(gd, traces, layout, {responsive: true, displayModeBar: false});
  renderCards();
}

function renderCards(){
  if (!selected.size){
    cards.innerHTML = "<p style='color:#999;font-size:13px'>Select a theme or sub-topic to see its details here.</p>";
    return;
  }
  var html = "";
  selected.forEach(function(key){
    var t = byKey(key); if (!t) return;
    if (key.charAt(0) === "g"){
      html += "<div class='card'>"
           +   "<h3><span class='sw' style='background:" + t.color + "'></span>" + esc(t.name) + "</h3>"
           +   "<div class='meta'>" + t.papers + " papers \\u00b7 " + t.subs.length + " sub-topics \\u00b7 peak " + t.peak_bucket + "s at " + (t.peak_share*100).toFixed(1) + "% share</div>"
           +   "<div class='members'>"
           +   t.subs.map(function(tid){
                 var s = SUBS[tid]; var on = selected.has(s.key) ? " on" : "";
                 return "<span class='member" + on + "' data-key='" + esc(s.key) + "'>" + esc(s.label) + " (" + s.papers + ")</span>";
               }).join("")
           +   "</div></div>";
    } else {
      html += "<div class='card'>"
           +   "<h3><span class='sw' style='background:" + t.color + "'></span>" + esc(t.label) + "</h3>"
           +   "<div class='kw'>" + esc(t.theme) + " \\u00b7 " + esc(t.top_words.join(", ")) + "</div>"
           +   "<div class='meta'>" + t.papers + " papers \\u00b7 peak " + t.peak_bucket + "s at " + (t.peak_share*100).toFixed(1) + "% share</div>";
      if (t.papers_list.length){
        html += "<ol>";
        t.papers_list.forEach(function(p){
          var link = p.u ? "<a href='" + esc(p.u) + "' target='_blank' rel='noopener'>" + esc(p.t) + "</a>" : esc(p.t);
          var cite = p.c ? ", " + p.c + " citations" : "";
          var auth = p.a ? "<br><span class='meta'>" + esc(p.a) + "</span>" : "";
          html += "<li>" + link + " <span class='meta'>(" + p.y + cite + ")</span>" + auth + "</li>";
        });
        html += "</ol>";
      }
      html += "</div>";
    }
  });
  cards.innerHTML = html;
  cards.querySelectorAll(".member").forEach(function(el){
    el.addEventListener("click", function(){ toggleSelect(el.dataset.key); });
  });
}

document.getElementById("expand-all").addEventListener("click", function(){
  THEMES.forEach(function(g){ expanded.add(g.name); }); renderBrowser();
});
document.getElementById("collapse-all").addEventListener("click", function(){
  expanded.clear(); renderBrowser();
});
document.getElementById("clear").addEventListener("click", function(){
  selected.clear(); renderBrowser(); redraw();
});

renderBrowser();
redraw();
"""


def plot(buckets, labels, themes, subs):
    fig = go.Figure()
    fig.update_layout(
        title="Pick a theme or sub-topic to plot its drift",
        xaxis=dict(title="5-year period"),
        yaxis=dict(title="Share of ICSE papers", tickformat=".0%"),
        height=520,
        template="plotly_white",
        margin=dict(t=50, l=60, r=20, b=70),
    )

    post = (
        _POST_SCRIPT.replace("__BUCKETS__", json.dumps(buckets))
        .replace("__LABELS__", json.dumps(labels))
        .replace("__THEMES__", json.dumps(themes, ensure_ascii=False).replace("</", "<\\/"))
        .replace("__SUBS__", json.dumps(subs, ensure_ascii=False).replace("</", "<\\/"))
    )

    dest = FIGURES_DIR / f"{NAME}.html"
    fig.write_html(str(dest), include_plotlyjs="cdn", post_script=post)
    log.info("  wrote %s", dest)


def main():
    buckets, labels, themes, subs = build()
    log.info(
        "ICSE drift browser: %d themes, %d sub-topics, %d buckets",
        len(themes),
        len(subs),
        len(buckets),
    )
    plot(buckets, labels, themes, subs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()