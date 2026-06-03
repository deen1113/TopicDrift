"""
topic_drift_search.py — Query-driven drift explorer.

WHAT IT DOES
  A search box over every topic's keywords. As the user types ("mobile", "llm",
  "agile"), matching topics light up in a list; clicking a topic adds its share-
  over-time curve to an overlay line chart, with a marker at its peak 5-year
  bucket. Multiple topics can be compared on the same axes; each selected topic
  also expands a small card below the plot showing its top three most-cited
  papers as links.

  Preset chips at the top replay the eight historical SE milestones from
  outputs/tables/topic_sanity_events.csv (Object-oriented, Agile, Mining repos,
  Cloud, Mobile, Deep learning, DevOps, LLMs) so the viz doubles as an
  interactive sanity-check tour: click a chip, see which topic the method
  associates with that event and when it actually peaked.

WHAT IT HOPES TO ANSWER
  "When did ICSE talk about X, and how much?" — for any X the user can name.
  Complements the streamgraph (top-15 only) and the treemap (decade-grouped)
  by giving the long-tail topics first-class access via search.

Reads:  data/processed/icse_topics.parquet, icse_topics_over_time.parquet,
        icse_paper_topics.parquet, data/interim/icse_enriched.parquet
Writes: outputs/figures/topic_drift_search.html
"""

import json

import plotly.graph_objects as go

from _common import FIGURES_DIR, load_paper_topics, load_topics, load_tot, short_label

NAME = "topic_drift_search"

# Each preset is (chip label, query string fed into the search box). Queries
# are space-separated tokens; a topic matches if ANY token appears in its top
# words or keyword label. Mirrors the events in topic_sanity_events.csv so the
# viz lines up with the paper's validation table.
MILESTONES = [
    ("Object-oriented (1990)", "object oriented"),
    ("Software process / CMM (1995)", "process cmm maturity"),
    ("Agile / Scrum (2001)", "agile scrum"),
    ("Mining repos (2004)", "mining repository repositories"),
    ("Cloud / SaaS (2008)", "cloud saas service services"),
    ("Mobile / Android (2010)", "mobile android apps"),
    ("Deep learning (2014)", "deep neural dnn learning"),
    ("DevOps (2015)", "devops continuous deployment"),
    ("LLM / code-gen (2022)", "llm llms language completion"),
]


def build():
    """Return (buckets, topic_index) where topic_index is keyed by topic_id."""
    tot = load_tot()
    topics = load_topics()
    pt = load_paper_topics()

    buckets = sorted(int(b) for b in tot["year_bucket"].unique())
    bucket_labels = [f"{b}s" for b in buckets]

    # nlargest twice would scan the whole frame per topic; group once instead.
    top_papers = {}
    for tid, grp in pt[pt["topic_id"] != -1].groupby("topic_id"):
        rows = grp.nlargest(3, "citation_count")
        top_papers[int(tid)] = [
            {
                "t": r["title"] or "(untitled)",
                "y": int(r["year"]),
                "c": int(r["citation_count"]),
                "u": r["paper_url"] or "",
            }
            for _, r in rows.iterrows()
        ]

    topic_index = {}
    for _, row in topics.iterrows():
        tid = int(row["topic_id"])
        if tid == -1:
            continue
        sub = tot[tot["topic_id"] == tid].set_index("year_bucket")
        shares = [float(sub["share"].get(b, 0.0)) for b in buckets]
        freqs = [int(sub["freq"].get(b, 0)) for b in buckets]
        peak_i = max(range(len(shares)), key=lambda i: shares[i])

        top_words = [str(w) for w in row["top_words"]]
        label = short_label(top_words, n=3)
        # Build a single lowercase search blob so JS can do substring matching
        # without re-tokenising on every keystroke.
        searchable = (" ".join(top_words) + " " + label).lower()

        topic_index[tid] = {
            "id": tid,
            "label": label,
            "top_words": top_words[:10],
            "shares": shares,
            "freqs": freqs,
            "peak_bucket": buckets[peak_i],
            "peak_share": shares[peak_i],
            "total_papers": int(row["size"]),
            "papers": top_papers.get(tid, []),
            "search": searchable,
        }

    return buckets, bucket_labels, topic_index


_POST_SCRIPT = """
var BUCKETS = __BUCKETS__;
var LABELS  = __LABELS__;
var TOPICS  = __TOPICS__;        // {topic_id: {label, top_words, shares, freqs, peak_bucket, peak_share, total_papers, papers, search}}
var MILESTONES = __MILESTONES__; // [[label, query], ...]

var gd = document.getElementById("{plot_id}");
var selected = new Set();
var mode = "share"; // or "freq"

var style = document.createElement("style");
style.textContent =
  "body{font-family:sans-serif;margin:0;background:#fafbfc;color:#222;transition:background .2s,color .2s}"
+ "#search-wrap{padding:14px 18px;border-bottom:1px solid #e6e9ec;background:#fff}"
+ "#search-wrap h1{margin:0 0 4px;font-size:18px}"
+ "#search-wrap p{margin:0 0 10px;color:#666;font-size:13px}"
+ "#search-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}"
+ "#q{flex:1;min-width:240px;padding:9px 12px;font-size:14px;border:1px solid #cdd3da;border-radius:6px;outline:none;background:#fff;color:#222}"
+ "#q:focus{border-color:#4a90d9;box-shadow:0 0 0 3px rgba(74,144,217,.15)}"
+ "#mode-toggle{display:flex;gap:0;border:1px solid #cdd3da;border-radius:6px;overflow:hidden}"
+ "#mode-toggle button{background:#fff;border:none;padding:8px 12px;font-size:13px;cursor:pointer;color:#555}"
+ "#mode-toggle button.active{background:#2a3f5f;color:#fff;font-weight:600}"
+ "#mode-toggle button:not(:last-child){border-right:1px solid #cdd3da}"
+ "#chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}"
+ ".chip{background:#eef3f8;border:1px solid #cdd9e5;color:#2a3f5f;padding:5px 11px;font-size:12px;border-radius:13px;cursor:pointer}"
+ ".chip:hover{background:#dde6f0}"
+ "#clear{background:#fff;border:1px solid #d8534f;color:#d8534f;padding:5px 11px;font-size:12px;border-radius:13px;cursor:pointer;margin-left:auto}"
+ "#clear:hover{background:#fdecec}"
+ "#main{display:grid;grid-template-columns:300px 1fr;gap:0;align-items:start}"
+ "#hits{max-height:520px;overflow-y:auto;border-right:1px solid #e6e9ec;background:#fff}"
+ "#hits .hit{padding:8px 12px;border-bottom:1px solid #f0f2f4;cursor:pointer;font-size:13px}"
+ "#hits .hit:hover{background:#f5f9fd}"
+ "#hits .hit.on{background:#e7f0fa;border-left:3px solid #2a3f5f;padding-left:9px}"
+ "#hits .lbl{font-weight:600;color:#222}"
+ "#hits .meta{color:#888;font-size:11px;margin-top:2px}"
+ "#hits .empty{padding:18px;color:#999;font-size:13px;text-align:center}"
+ "#plot{min-height:520px}"
+ "#cards{padding:14px 18px;background:#fff;border-top:1px solid #e6e9ec}"
+ ".card{margin-bottom:14px;padding:11px 14px;border:1px solid #e6e9ec;border-radius:8px;background:#fafbfc}"
+ ".card h3{margin:0 0 4px;font-size:14px}"
+ ".card .kw{color:#888;font-size:12px;font-style:italic;margin-bottom:6px}"
+ ".card ol{margin:6px 0 0;padding-left:22px;font-size:13px}"
+ ".card a{color:#2a3f5f;text-decoration:none}"
+ ".card a:hover{text-decoration:underline}"
+ ".card .meta{color:#999;font-size:12px}"
// Dark-mode overrides: the host page (docs/index.html) toggles document.body.dark
// on this iframe via postMessage. Every coloured rule above gets a counterpart
// here so the viz follows the rest of the site.
+ "body.dark{background:#1e1e1e;color:#e6e6e6}"
+ "body.dark #search-wrap{background:#252525;border-bottom-color:#3a3a3a}"
+ "body.dark #search-wrap p{color:#aaa}"
+ "body.dark #q{background:#2a2a2a;border-color:#555;color:#e6e6e6}"
+ "body.dark #q:focus{border-color:#4a90d9}"
+ "body.dark #mode-toggle{border-color:#555}"
+ "body.dark #mode-toggle button{background:#2a2a2a;color:#ccc}"
+ "body.dark #mode-toggle button.active{background:#4a90d9;color:#fff}"
+ "body.dark #mode-toggle button:not(:last-child){border-right-color:#555}"
+ "body.dark .chip{background:#2a3f5f;border-color:#3f5577;color:#cfe1f4}"
+ "body.dark .chip:hover{background:#34527a}"
+ "body.dark #clear{background:#2a2a2a}"
+ "body.dark #clear:hover{background:#3a2424}"
+ "body.dark #hits{background:#252525;border-right-color:#3a3a3a}"
+ "body.dark #hits .hit{border-bottom-color:#303030}"
+ "body.dark #hits .hit:hover{background:#2a2a2a}"
+ "body.dark #hits .hit.on{background:#2a3f5f;border-left-color:#4a90d9}"
+ "body.dark #hits .lbl{color:#e6e6e6}"
+ "body.dark #hits .meta,body.dark #hits .empty{color:#999}"
+ "body.dark #cards{background:#252525;border-top-color:#3a3a3a}"
+ "body.dark .card{background:#2a2a2a;border-color:#3a3a3a}"
+ "body.dark .card .kw,body.dark .card .meta{color:#888}"
+ "body.dark .card a{color:#6db3f2}";
document.head.appendChild(style);

// Dark mode is owned by the host page; when this viz is embedded as an iframe
// the parent posts {type:'td-dark', on:bool} on toggle. Apply both the CSS
// class (drives every rule above) and Plotly's paper/plot bg + font colour.
var darkOn = false;
function applyDark(on){
  darkOn = !!on;
  document.body.classList.toggle("dark", darkOn);
  // relayout updates the existing figure; the next redraw() will pick up the
  // matching layout colours via the same darkOn flag.
  Plotly.relayout(gd, darkOn
    ? {paper_bgcolor:"#1e1e1e", plot_bgcolor:"#1e1e1e", "font.color":"#e6e6e6",
       "xaxis.gridcolor":"#3a3a3a", "yaxis.gridcolor":"#3a3a3a"}
    : {paper_bgcolor:"white", plot_bgcolor:"white", "font.color":"#2a3f5f",
       "xaxis.gridcolor":"#eee", "yaxis.gridcolor":"#eee"});
}
window.addEventListener("message", function(ev){
  if (ev.data && ev.data.type === "td-dark"){ applyDark(ev.data.on); }
});

// Build the search UI above the (initially empty) plotly figure.
var wrap = document.createElement("div");
wrap.id = "search-wrap";
wrap.innerHTML =
    "<h1>Topic Drift Search</h1>"
  + "<p>Type a keyword (try <i>mobile</i>, <i>llm</i>, <i>testing</i>) or click a milestone chip. "
  + "Pick topics from the list to overlay their drift curves.</p>"
  + "<div id='search-row'>"
  +   "<input id='q' placeholder='Search topic keywords...' autocomplete='off'/>"
  +   "<div id='mode-toggle'>"
  +     "<button data-mode='share' class='active'>Share</button>"
  +     "<button data-mode='freq'>Paper count</button>"
  +   "</div>"
  +   "<button id='clear'>Clear selection</button>"
  + "</div>"
  + "<div id='chips'></div>";
gd.parentNode.insertBefore(wrap, gd);

var chips = document.getElementById("chips");
MILESTONES.forEach(function(m){
  var b = document.createElement("button");
  b.className = "chip"; b.type = "button"; b.textContent = m[0];
  b.addEventListener("click", function(){
    document.getElementById("q").value = m[1];
    runSearch(m[1], true);
  });
  chips.appendChild(b);
});

// Layout: hits list on the left, plot on the right, cards below.
var main = document.createElement("div"); main.id = "main";
var hits = document.createElement("div"); hits.id = "hits";
hits.innerHTML = "<div class='empty'>Type above or pick a milestone chip.</div>";
gd.parentNode.insertBefore(main, gd);
main.appendChild(hits);
main.appendChild(gd); gd.id = "plot";

var cards = document.createElement("div"); cards.id = "cards";
cards.innerHTML = "<p style='color:#999;font-size:13px'>Selected topics will list their top-cited papers here.</p>";
main.parentNode.appendChild(cards);

function esc(s){ return (s+"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function search(query){
  var q = (query || "").trim().toLowerCase();
  if (!q) return [];
  // Tokenise; a topic matches if ANY token is a substring of its search blob.
  var toks = q.split(/\\s+/).filter(Boolean);
  var matches = [];
  for (var tid in TOPICS){
    var t = TOPICS[tid];
    var hit = false;
    for (var i = 0; i < toks.length; i++){
      if (t.search.indexOf(toks[i]) !== -1){ hit = true; break; }
    }
    if (hit) matches.push(t);
  }
  matches.sort(function(a, b){ return b.total_papers - a.total_papers; });
  return matches.slice(0, 40);
}

function renderHits(matches){
  if (!matches.length){
    hits.innerHTML = "<div class='empty'>No matching topics. Try a broader term.</div>";
    return;
  }
  var html = "";
  matches.forEach(function(t){
    var on = selected.has(t.id) ? " on" : "";
    html += "<div class='hit" + on + "' data-tid='" + t.id + "'>"
         +    "<div class='lbl'>" + esc(t.label) + "</div>"
         +    "<div class='meta'>" + t.total_papers + " papers · peak " + t.peak_bucket + "s (" + (t.peak_share*100).toFixed(1) + "%)</div>"
         +  "</div>";
  });
  hits.innerHTML = html;
  hits.querySelectorAll(".hit").forEach(function(el){
    el.addEventListener("click", function(){ toggle(parseInt(el.dataset.tid, 10)); });
  });
}

function toggle(tid){
  if (selected.has(tid)) selected.delete(tid);
  else selected.add(tid);
  // Refresh row highlight without losing the current search results.
  hits.querySelectorAll(".hit").forEach(function(el){
    el.classList.toggle("on", parseInt(el.dataset.tid, 10) === tid ? selected.has(tid) : selected.has(parseInt(el.dataset.tid, 10)));
  });
  redraw();
}

function runSearch(q, autoselect){
  var matches = search(q);
  renderHits(matches);
  // For milestone chips: auto-select the single best (highest-volume) match so
  // the user sees a curve immediately without an extra click.
  if (autoselect && matches.length){
    selected.clear();
    selected.add(matches[0].id);
    // Repaint hit highlight after the auto-pick
    renderHits(matches);
    redraw();
  }
}

function redraw(){
  var traces = [];
  selected.forEach(function(tid){
    var t = TOPICS[tid];
    var ys = mode === "share" ? t.shares : t.freqs;
    var customdata = mode === "share" ? t.freqs : t.shares;
    var fmt = mode === "share" ? "%{y:.1%}" : "%{y}";
    var other = mode === "share" ? "papers: %{customdata}" : "share: %{customdata:.1%}";
    traces.push({
      x: LABELS, y: ys, customdata: customdata,
      mode: "lines+markers", name: t.label, type: "scatter",
      line: {width: 2.5}, marker: {size: 6},
      hovertemplate: "<b>" + t.label + "</b><br>%{x}: " + fmt + "<br>" + other + "<extra></extra>"
    });
    // Peak marker: bigger filled dot at the bucket where this topic peaked.
    // Outline + label colour flip with darkOn so the star reads on either bg.
    var peakIdx = BUCKETS.indexOf(t.peak_bucket);
    if (peakIdx !== -1){
      traces.push({
        x: [LABELS[peakIdx]], y: [ys[peakIdx]],
        mode: "markers+text", type: "scatter",
        marker: {size: 14, symbol: "star", color: "rgba(0,0,0,0)",
                 line: {color: darkOn ? "#e6e6e6" : "#222", width: 1.5}},
        text: ["peak"], textposition: "top center",
        textfont: {size: 10, color: darkOn ? "#bbb" : "#555"},
        showlegend: false, hoverinfo: "skip"
      });
    }
  });
  // Plotly.react would reset paper_bgcolor/font.color to the template defaults
  // every time, so we fold the current dark state into the layout each redraw.
  var grid = darkOn ? "#3a3a3a" : "#eee";
  var layout = {
    title: selected.size ? "Drift curves — " + selected.size + " topic" + (selected.size > 1 ? "s" : "") + " selected" : "Pick a topic to plot its drift",
    xaxis: {title: "5-year bucket", gridcolor: grid},
    yaxis: {title: mode === "share" ? "Share of papers in bucket" : "Papers in bucket",
            tickformat: mode === "share" ? ".0%" : ",d", rangemode: "tozero", gridcolor: grid},
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
    cards.innerHTML = "<p style='color:#999;font-size:13px'>Selected topics will list their top-cited papers here.</p>";
    return;
  }
  var html = "";
  selected.forEach(function(tid){
    var t = TOPICS[tid];
    html += "<div class='card'>"
         +   "<h3>" + esc(t.label) + "</h3>"
         +   "<div class='kw'>" + esc(t.top_words.join(", ")) + "</div>"
         +   "<div class='meta'>" + t.total_papers + " papers · peak " + t.peak_bucket + "s at " + (t.peak_share*100).toFixed(1) + "% share</div>";
    if (t.papers.length){
      html += "<ol>";
      t.papers.forEach(function(p){
        var link = p.u ? "<a href='" + esc(p.u) + "' target='_blank' rel='noopener'>" + esc(p.t) + "</a>" : esc(p.t);
        html += "<li>" + link + " <span class='meta'>(" + p.y + ", " + p.c + " citations)</span></li>";
      });
      html += "</ol>";
    }
    html += "</div>";
  });
  cards.innerHTML = html;
}

// Wire up search-on-keystroke (debounced) and the mode toggle.
var debounce;
document.getElementById("q").addEventListener("input", function(e){
  clearTimeout(debounce);
  debounce = setTimeout(function(){ runSearch(e.target.value, false); }, 120);
});
document.querySelectorAll("#mode-toggle button").forEach(function(b){
  b.addEventListener("click", function(){
    document.querySelectorAll("#mode-toggle button").forEach(function(x){ x.classList.remove("active"); });
    b.classList.add("active");
    mode = b.dataset.mode;
    redraw();
  });
});
document.getElementById("clear").addEventListener("click", function(){
  selected.clear();
  document.getElementById("q").value = "";
  hits.innerHTML = "<div class='empty'>Type above or pick a milestone chip.</div>";
  redraw();
});

// Initial: nothing selected, empty plot with hint title.
redraw();
"""


def plot(buckets, bucket_labels, topic_index):
    # Start with an empty figure; everything is drawn client-side from the
    # embedded JSON via Plotly.react inside the post-script.
    fig = go.Figure()
    fig.update_layout(
        title="Pick a topic to plot its drift",
        xaxis=dict(title="5-year bucket"),
        yaxis=dict(title="Share of papers in bucket", tickformat=".0%"),
        height=520,
        template="plotly_white",
        margin=dict(t=50, l=60, r=20, b=70),
    )

    buckets_json = json.dumps(buckets)
    labels_json = json.dumps(bucket_labels)
    topics_json = json.dumps(topic_index, ensure_ascii=False).replace("</", "<\\/")
    milestones_json = json.dumps(MILESTONES)

    post = (
        _POST_SCRIPT.replace("__BUCKETS__", buckets_json)
        .replace("__LABELS__", labels_json)
        .replace("__TOPICS__", topics_json)
        .replace("__MILESTONES__", milestones_json)
    )

    dest = FIGURES_DIR / f"{NAME}.html"
    fig.write_html(str(dest), include_plotlyjs="cdn", post_script=post)
    print(f"  wrote {dest}")


def main():
    buckets, bucket_labels, topic_index = build()
    print(f"Search index: {len(topic_index)} topics across {len(buckets)} buckets")
    plot(buckets, bucket_labels, topic_index)


if __name__ == "__main__":
    main()
