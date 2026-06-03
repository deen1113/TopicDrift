"""
topic_treemap.py — Hierarchical decade → topic treemap with a click-to-list panel.

WHAT IT DOES
  Lays out a scope's corpus as a treemap: root → decade → overarching theme →
  topic. Each tile's area is the number of papers.

  Colour is the overarching theme: every subtopic takes its theme's colour, so a
  theme reads as one coloured block (tile text auto-set to black/white for
  contrast). Mode toggles re-colour the topic tiles by median citations (an
  impact heatmap), growth-vs-previous-decade, or whether a topic is new this
  decade. Hover shows the key stats; clicking a topic tile lists its papers below.

  THREE SCOPES, TWO DATA SOURCES
    • icse  — the ICSE-only fit (icse_* tables). Fully enriched: citation counts,
              authors and paper URLs are present, so every colour mode and every
              stat card is available. This is the original, richest treemap.
    • top10 / all — the global multi-conference fit (conf_* tables). These have
              titles + abstracts but NO citation/author/URL enrichment, so the
              impact colour mode, the citation stat cards, author lists and paper
              links are dropped. Identity / Growth / Emerging modes and a
              title+year paper list all still work (they are share-based).

  Everything that needs citations/authors/links is guarded by `has_cites` on
  both the Python and the JS side, degrading gracefully when the data is absent.

Reads (icse):  data/processed/icse_paper_topics.parquet (+ icse_enriched join),
               data/processed/icse_topics.parquet
Reads (conf):  data/processed/conf_paper_topics.parquet, conf_enriched.parquet,
               conf_topics.parquet
Writes: outputs/figures/topic_treemap_{scope}.html  (one per scope)
"""
import json
import math
import re
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

from topicdrift.visualization._common import (
    PROCESSED_DIR, FIGURES_DIR, SCOPE_TITLES,
    load_paper_topics, load_topics, scope_filter, short_label, topic_labels,
)

NAME = "topic_treemap"

# For the conf scopes (no citations to rank by) the per topic×decade paper list
# is capped to the most recent CAP papers to keep the embedded JSON — and the
# page — a sane size; a "+N more" note flags the truncation.
CONF_LIST_CAP = 200

# Neutral fill for the structural (root + decade) tiles so topic colours pop.
# A dark variant is swapped in client-side when dark mode is on.
NEUTRAL_BG = "#e9ecef"
NEUTRAL_FG = "#343a40"
DARK_NEUTRAL_BG = "#3a4046"
DARK_NEUTRAL_FG = "#e6e6e6"

# Per-scope root-tile blurb shown when you click the root. ICSE keeps the
# conference write-up; the global scopes describe the venue filter.
ROOT_BLURBS = {
    "icse": (
        "The International Conference on Software Engineering (ICSE) is the flagship "
        "conference of the software engineering research community, held annually "
        "since 1975 and sponsored by ACM SIGSOFT and IEEE TCSE. This treemap maps its "
        "research paper corpus: the first level is the publication decade and the "
        "second is the topic within that decade. Tile area is the number of papers and "
        "tile colour identifies the topic (or, in impact mode, its median citations). "
        "Click a decade to see what dominated that era, or a topic tile to drill into "
        "its stats and papers."
    ),
    "top10": (
        "The ten flagship software-engineering and programming-languages venues — "
        "ICSE, ESEC/FSE, ASE, ISSTA, ICSME, MSR, RE, OOPSLA, PLDI and POPL — sharing "
        "the one global topic space used across the site. This treemap maps their "
        "pooled corpus: the first level is the publication decade and the second is "
        "the topic within that decade. Tile area is the number of papers and tile "
        "colour identifies the topic. Click a decade to see what dominated that era, "
        "or a topic tile to drill into its papers."
    ),
    "all": (
        "Every DBLP conference with usable abstracts (≥50 papers and ≥50% abstract "
        "coverage) — 2,000+ venues spanning all of computer science, on the one global "
        "topic space used across the site. This treemap maps the whole corpus: the "
        "first level is the publication decade and the second is the topic within that "
        "decade. Tile area is the number of papers and tile colour identifies the "
        "topic. Click a decade to see what dominated that era, or a topic tile to drill "
        "into its papers."
    ),
}


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


# ── Per-scope data sources ─────────────────────────────────────────────────
# A "source" bundles everything build() needs: the per-paper table, the topic
# title/keyword map, the root label/blurb, and whether citation enrichment is
# available. ICSE draws on the fully-enriched icse_* fit; the global scopes draw
# on the conf_* fit, which carries titles but no citations/authors/links.

def _icse_titles() -> dict[int, dict[str, str]]:
    """{topic_id: {'title', 'keywords'}} for the ICSE fit (llm_label + top words)."""
    keywords = topic_labels()  # {topic_id: 'word1 · word2 · word3'}
    topics = load_topics()
    has_llm = "llm_label" in topics.columns
    out: dict[int, dict[str, str]] = {}
    for _, r in topics.iterrows():
        tid = int(r["topic_id"])
        if tid not in keywords:  # skips the -1 outlier topic
            continue
        kw = keywords[tid]
        llm = str(r["llm_label"]).strip() if has_llm and pd.notna(r["llm_label"]) else ""
        out[tid] = {"title": llm or kw, "keywords": kw}
    return out


def _conf_titles() -> dict[int, dict[str, str]]:
    """{topic_id: {'title', 'keywords'}} for the global conf fit."""
    topics = pd.read_parquet(PROCESSED_DIR / "conf_topics.parquet")
    out: dict[int, dict[str, str]] = {}
    for _, r in topics.iterrows():
        tid = int(r["topic_id"])
        if tid == -1:
            continue
        kw = short_label(r["top_words"])
        llm = str(r["llm_label"]).strip() if pd.notna(r["llm_label"]) else ""
        out[tid] = {"title": llm or kw, "keywords": kw}
    return out


def _topic_groups(scope: str) -> tuple[dict[int, str], dict[str, str]]:
    """({topic_id: theme}, {theme: colour}) for the scope's fit — the overarching
    grouping used as the treemap's middle (theme) level."""
    name = "icse_topics" if scope == "icse" else "conf_topics"
    t = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    tg = {int(r.topic_id): str(r.group) for r in t.itertuples() if pd.notna(r.group)}
    gc = {str(r.group): str(r.group_color) for r in t.itertuples()
          if pd.notna(r.group) and pd.notna(r.group_color)}
    return tg, gc


def scope_source(scope: str) -> dict:
    """Bundle the per-paper table, titles, root label/blurb and has_cites flag."""
    root_label = SCOPE_TITLES.get(scope, scope)
    blurb = ROOT_BLURBS.get(scope, "")
    if scope == "icse":
        # Fully enriched ICSE fit: title, citation_count, authors, paper_url.
        pt = load_paper_topics()
        return {"pt": pt, "titles": _icse_titles(), "root_label": root_label,
                "blurb": blurb, "has_cites": True}
    # Global conf fit: titles only (no citation/author/url enrichment).
    pt = scope_filter(load_conf_paper_topics(), scope)
    titles = pd.read_parquet(PROCESSED_DIR / "conf_enriched.parquet",
                             columns=["dblp_key", "title"])
    pt = pt.merge(titles, on="dblp_key", how="left")
    return {"pt": pt, "titles": _conf_titles(), "root_label": root_label,
            "blurb": blurb, "has_cites": False}


def load_conf_paper_topics() -> pd.DataFrame:
    """conf_paper_topics with the scope/venue column kept for filtering."""
    return pd.read_parquet(PROCESSED_DIR / "conf_paper_topics.parquet")


def _text_on(color: str) -> str:
    """Black or white text, whichever reads on the given background. Accepts
    both '#rrggbb' (identity hues) and 'rgb(r, g, b)' (colorscale samples)."""
    if color.startswith("#"):
        r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    else:
        r, g, b = (float(v) for v in re.findall(r"[\d.]+", color)[:3])
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1a1a1a" if luminance > 140 else "#ffffff"


# Colours for the growth view: "new" topics get a distinct blue (emerged this
# decade), the first decade / no-prior-decade case is neutral grey, and the rest
# sit on a red→green diverging scale keyed on log2 fold-change of share.
GROWTH_NEW = "#3d7dd8"
GROWTH_NA = "#d3d7dc"
GROWTH_FLAT_PCT = 5.0  # |Δ| below this counts as "flat", not up/down
EMERGE_GREY = "#cdd2d8"  # non-new topics in the emerging view


def _prev_decade(decade: str) -> str:
    """'2010s' → '2000s'."""
    return f"{int(decade[:-1]) - 10}s"


def _growth_color(state: str, l2fc: float) -> str:
    """Diverging colour for the growth view. `l2fc` is log2(share / prev_share),
    clamped to ±2 (a quartering … a quadrupling) so a halving and a doubling read
    with equal weight on the scale."""
    if state in ("na", "new"):
        return GROWTH_NA
    x = max(-2.0, min(2.0, l2fc))
    return sample_colorscale("RdYlGn", (x + 2.0) / 4.0)[0]


def _add_growth(agg):
    """Add per topic×decade growth columns to `agg`, comparing each topic's
    SHARE of its decade to its share in the previous decade (raw counts would be
    swamped by the corpus growing over time).

    Columns: share, prev_decade, growth_state ('up'/'down'/'flat'/'new'/'na'),
    growth_pct (NaN when new/na), l2fc (NaN when new/na)."""
    agg = agg.copy()
    agg["share"] = agg["papers"] / agg.groupby("decade")["papers"].transform("sum")
    share = {(r.decade, int(r.topic_id)): r.share for r in agg.itertuples()}
    have = set(agg["decade"])

    prevs, states, pcts, l2s = [], [], [], []
    for r in agg.itertuples():
        pdec = _prev_decade(r.decade)
        prevs.append(pdec)
        prev = share.get((pdec, int(r.topic_id)), 0.0)
        if pdec not in have:                      # earliest decade — nothing to compare
            states.append("na"); pcts.append(float("nan")); l2s.append(float("nan"))
        elif prev <= 0.0:                         # topic absent last decade — emerged
            states.append("new"); pcts.append(float("nan")); l2s.append(float("nan"))
        else:
            pct = (r.share / prev - 1.0) * 100.0
            states.append("up" if pct > GROWTH_FLAT_PCT else
                          "down" if pct < -GROWTH_FLAT_PCT else "flat")
            pcts.append(pct); l2s.append(math.log2(r.share / prev))
    agg["prev_decade"] = prevs
    agg["growth_state"] = states
    agg["growth_pct"] = pcts
    agg["l2fc"] = l2s
    return agg


def _stats(grp, has_cites) -> dict:
    """Per topic×decade headline stats for the click panel. With citations the
    full impact suite is computed (grp is sorted citations-descending so the
    flagship paper is the first row); without, just count + year range."""
    n = len(grp)
    out = {"n": n, "ymin": int(grp["year"].min()), "ymax": int(grp["year"].max())}
    if not has_cites:
        return out
    cites = grp["citation_count"].astype(int).tolist()  # descending
    h = 0
    for i, c in enumerate(cites, 1):
        if c >= i:
            h = i
        else:
            break
    authors = Counter()
    for arr in grp["authors"]:
        for a in (list(arr) if arr is not None else []):
            authors[_clean_author(a)] += 1
    out.update({
        "med": round(grp["citation_count"].median()),
        "mean": round(grp["citation_count"].mean(), 1),
        "tot": int(grp["citation_count"].sum()),
        "h": h,
        "uncpct": round(sum(c == 0 for c in cites) / n * 100),
        "auth": authors.most_common(3),
    })
    return out


def _meta(agg, pt, has_cites, root_label, blurb) -> dict:
    """Descriptions for the structural tiles: the corpus (root) and each decade.
    Shown when you click a decade/root, vs. the per-topic stats shown when you
    drill into a topic tile. Citation-derived fields are included only when the
    scope carries citation data."""
    corpus_total = int(agg["papers"].sum())

    by_topic = (agg.groupby(["topic_id", "topic"], as_index=False)["papers"].sum()
                .nlargest(6, "papers"))
    root = {
        "title": root_label,
        "blurb": blurb,
        "n_papers": corpus_total,
        "n_topics": int(agg["topic_id"].nunique()),
        "ymin": int(pt["year"].min()),
        "ymax": int(pt["year"].max()),
        "n_decades": int(agg["decade"].nunique()),
        "busiest_decade": agg.groupby("decade")["papers"].sum().idxmax(),
        "top_topics": [[r.topic, int(r.papers)] for r in by_topic.itertuples()],
    }
    if has_cites:
        by_cited = (agg.groupby("topic", as_index=False)["total_citations"].sum()
                    .nlargest(1, "total_citations"))
        flag = pt.loc[pt["citation_count"].idxmax()]
        root.update({
            "total_citations": int(pt["citation_count"].sum()),
            "median_citations": round(pt["citation_count"].median()),
            "top_cited_topic": [by_cited.iloc[0]["topic"],
                                int(by_cited.iloc[0]["total_citations"])],
            "flagship": {"t": flag["title"] or "(untitled)", "u": flag["paper_url"] or "",
                         "a": _authors_str(flag["authors"]),
                         "y": int(flag["year"]), "c": int(flag["citation_count"])},
        })

    dec_median = (pt.groupby("decade")["citation_count"].median() if has_cites else None)

    # Per-decade topic membership + sizes, used to compute what faded away.
    members = {d: dict(zip(s["topic_id"].astype(int), s["topic"]))
               for d, s in agg.groupby("decade")}
    sizes = {(d, int(r.topic_id)): int(r.papers)
             for d, s in agg.groupby("decade") for r in s.itertuples()}

    decades = {}
    for decade in sorted(agg["decade"].unique()):
        sub = agg[agg["decade"] == decade]
        total = int(sub["papers"].sum())
        prevalent = sub.nlargest(5, "papers")
        emerging = sub[sub["growth_state"] == "new"].nlargest(6, "papers")
        up = sub[sub["growth_state"] == "up"].nlargest(1, "growth_pct")
        down = sub[sub["growth_state"] == "down"].nsmallest(1, "growth_pct")

        pdec = _prev_decade(decade)
        faded_ids = set(members.get(pdec, {})) - set(members.get(decade, {}))
        faded = sorted(faded_ids, key=lambda t: sizes.get((pdec, t), 0), reverse=True)[:6]

        entry = {
            "n_papers": total,
            "n_topics": int(len(sub)),
            "pct_corpus": round(total / corpus_total * 100, 1),
            "top_topics": [[r.topic, int(r.papers), round(r.papers / total * 100)]
                           for r in prevalent.itertuples()],
            "emerging": [r.topic for r in emerging.itertuples()],
            "faded": [members[pdec][t] for t in faded],
            "rising": [up.iloc[0]["topic"], round(up.iloc[0]["growth_pct"])] if len(up) else None,
            "falling": [down.iloc[0]["topic"], round(down.iloc[0]["growth_pct"])] if len(down) else None,
        }
        if has_cites:
            cited = sub.nlargest(3, "total_citations")
            entry["median_cites"] = round(dec_median.get(decade, 0))
            entry["top_cited"] = [[r.topic, int(r.total_citations)] for r in cited.itertuples()]
        decades[decade] = entry
    return {"root": root, "decades": decades}


def _theme_meta(agg) -> dict:
    """Per decade×theme summary for the click panel when a theme tile is clicked:
    the theme's size, its share of the decade, and the topics it contains."""
    dec_totals = agg.groupby("decade")["papers"].sum()
    out = {}
    for (decade, theme), g in agg.groupby(["decade", "theme"]):
        tp = int(g["papers"].sum())
        tops = g.sort_values("papers", ascending=False)
        out[f"{decade}||{theme}"] = {
            "name": theme,
            "n_papers": tp,
            "n_topics": int(len(g)),
            "pct_decade": round(tp / dec_totals[decade] * 100, 1),
            "topics": [[r.topic, int(r.papers)] for r in tops.itertuples()],
        }
    return out


def build(scope: str):
    """Return (tile aggregate, panel index, meta, group_colors, has_cites).

    panel index: {'<decade>||<topic_id>': {name, stats, rows}} for topic tiles.
    meta: {'root', 'decades', 'themes'} — the structural-tile descriptions, where
    themes is {'<decade>||<theme>': {...}} for the theme level."""
    src = scope_source(scope)
    pt, titles, has_cites = src["pt"], src["titles"], src["has_cites"]
    groups, group_colors = _topic_groups(scope)
    pt = pt[pt["topic_id"].isin(titles)].copy()
    pt["decade"] = (pt["year"] // 10 * 10).astype(int).astype(str) + "s"
    pt["topic"] = pt["topic_id"].map(lambda t: titles[t]["title"])
    pt["keywords"] = pt["topic_id"].map(lambda t: titles[t]["keywords"])
    pt["theme"] = pt["topic_id"].map(groups).fillna("Other")

    # theme and keywords are both functionally determined by topic_id, so adding
    # them to the grouping is a no-op for the aggregation but carries the
    # overarching theme (the treemap's middle level) and the hover text through.
    agg_spec = {"papers": ("dblp_key", "size")}
    if has_cites:
        agg_spec.update(mean_citations=("citation_count", "mean"),
                        median_citations=("citation_count", "median"),
                        total_citations=("citation_count", "sum"))
    agg = pt.groupby(["decade", "theme", "topic_id", "topic", "keywords"]).agg(**agg_spec).reset_index()
    agg = _add_growth(agg)

    # Growth, keyed by (decade, topic_id), for the click panel's growth card.
    growth = {(r.decade, int(r.topic_id)):
              {"state": r.growth_state, "prev": r.prev_decade,
               "pct": None if math.isnan(r.growth_pct) else round(r.growth_pct)}
              for r in agg.itertuples()}

    # Panel index keyed by topic_id (not the keyword label) so two topics that
    # happen to share top words can't collide. With citations, papers are sorted
    # citations-descending and the whole list is kept; without, by year-descending
    # and capped (no citation signal to rank or trim by).
    sort_col = "citation_count" if has_cites else "year"
    papers: dict[str, dict] = {}
    for (decade, tid), grp in pt.sort_values(sort_col, ascending=False).groupby(["decade", "topic_id"]):
        listed = grp if has_cites else grp.head(CONF_LIST_CAP)
        rows = []
        for r in listed.itertuples():
            row = {"t": r.title or "(untitled)", "y": int(r.year)}
            if has_cites:
                row.update(a=_authors_str(r.authors), u=r.paper_url or "",
                           c=int(r.citation_count))
            rows.append(row)
        entry = {
            "name": titles[tid]["title"],
            "kw": titles[tid]["keywords"],
            "stats": _stats(grp, has_cites),
            "growth": growth.get((decade, int(tid))),
            "rows": rows,
        }
        if not has_cites and len(grp) > CONF_LIST_CAP:
            entry["more"] = int(len(grp) - CONF_LIST_CAP)
        papers[f"{decade}||{tid}"] = entry
    meta = _meta(agg, pt, has_cites, src["root_label"], src["blurb"])
    meta["themes"] = _theme_meta(agg)
    return agg, papers, meta, group_colors, has_cites


# JS injected into the HTML: builds a list panel under the treemap and fills it
# when a topic tile is clicked. {plot_id} is substituted by Plotly (literal
# replace, so the JSON braces below are safe). Citation-derived fields are
# null-checked so the same panel code serves both the enriched (ICSE) and the
# citation-free (conf) scopes.
_POST_SCRIPT = """
var DATA = __DATA__;
var META = __META__;
var THEMES = __THEMES__;      // {'<decade>||<theme>': {...}} for theme-tile clicks
var LEGEND = __LEGEND__;
var MODE_DEFS = __MODES__;   // [{label,key,bg,fg}] colour modes for the button bar
var STRUCT = __STRUCT__;     // indices of structural (root/decade) tiles
var NC = __NEUTRAL__;        // {lbg,lfg,dbg,dfg} neutral colours, light & dark
var gd = document.getElementById("{plot_id}");

var style = document.createElement("style");
style.textContent =
  "body{transition:background .2s,color .2s}"
+ "#modebar{display:flex;flex-wrap:wrap;gap:8px;padding:10px 12px;font-family:sans-serif;border-bottom:1px solid #eceff2;margin-bottom:4px}"
+ "body.dark #modebar{border-bottom-color:#3a3a3a}"
+ ".modebtn{background:#fff;border:1px solid #bcccdc;color:#2a3f5f;padding:6px 12px;font-size:13px;border-radius:6px;cursor:pointer;transition:background .12s,border-color .12s}"
+ ".modebtn:hover{background:#eef3f8}"
+ ".modebtn.active{background:#2a3f5f;color:#fff;border-color:#2a3f5f;font-weight:600}"
+ "body.dark #modebar .modebtn{background:#2a2a2a;border-color:#555;color:#e0e0e0}"
+ "body.dark #modebar .modebtn:hover{background:#3a3a3a}"
+ "body.dark #modebar .modebtn.active{background:#4a90d9;border-color:#4a90d9;color:#fff}"
+ ".statcard{position:relative;flex:0 0 auto;min-width:84px;background:#f4f5f7;border-radius:8px;padding:8px 12px;text-align:center;cursor:help}"
+ ".statcard .tip{visibility:hidden;opacity:0;transition:opacity .12s;position:absolute;bottom:calc(100% + 8px);left:50%;transform:translateX(-50%);"
+ "background:#222;color:#fff;padding:7px 10px;border-radius:6px;width:200px;font-size:.75em;line-height:1.35;text-align:left;text-transform:none;letter-spacing:0;z-index:20;box-shadow:0 2px 8px rgba(0,0,0,.25)}"
+ ".statcard .tip::after{content:'';position:absolute;top:100%;left:50%;transform:translateX(-50%);border:6px solid transparent;border-top-color:#222}"
+ ".statcard:hover .tip{visibility:visible;opacity:1}"
+ ".switch{position:relative;display:inline-block;width:42px;height:22px}"
+ ".switch input{opacity:0;width:0;height:0}"
+ ".sw{position:absolute;cursor:pointer;inset:0;background:#ccc;border-radius:22px;transition:.2s}"
+ ".sw:before{content:'';position:absolute;height:16px;width:16px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.2s}"
+ ".switch input:checked + .sw{background:#4a90d9}"
+ ".switch input:checked + .sw:before{transform:translateX(20px)}"
+ "body.dark{background:#1e1e1e;color:#e6e6e6}"
+ "body.dark #drift-panel,body.dark #drift-panel h2{color:#e6e6e6!important}"
+ "body.dark #drift-panel p,body.dark #drift-panel span{color:#b3b3b3!important}"
+ "body.dark #drift-panel a{color:#6db3f2!important}"
+ "body.dark #drift-panel .statcard{background:#2b2b2b!important}"
+ "body.dark #drift-panel .statcard>div:first-child{color:#f0f0f0!important}"
+ "body.dark #leg-panel{background:#262626!important;border-color:#454545!important}"
+ "body.dark #leg-handle{background:#1c1c1c!important;border-color:#454545!important}"
+ "body.dark #leg,body.dark #leg b,body.dark #leg span,body.dark #leg div{color:#e8e8e8!important}";
document.head.appendChild(style);

var panel = document.createElement("div");
panel.id = "drift-panel";
panel.style.cssText = "font-family:sans-serif;max-width:1100px;margin:18px auto;padding:0 14px;color:#222";
panel.innerHTML = "<p style='color:#777'>Click a topic tile above to list its papers.</p>";
gd.parentNode.insertBefore(panel, gd.nextSibling);

var darkOn = false, applyingStruct = false;
function applyStruct(){
  if (!STRUCT || !STRUCT.length) return;
  var d0 = gd.data[0];
  var cols = (d0.marker && d0.marker.colors) ? d0.marker.colors.slice() : [];
  if (!cols.length) return;
  var fc = (d0.textfont && Array.isArray(d0.textfont.color)) ? d0.textfont.color.slice() : null;
  STRUCT.forEach(function(i){
    cols[i] = darkOn ? NC.dbg : NC.lbg;
    if (fc) fc[i] = darkOn ? NC.dfg : NC.lfg;
  });
  applyingStruct = true;
  Plotly.restyle(gd, fc ? {"marker.colors":[cols], "textfont.color":[fc]} : {"marker.colors":[cols]}, [0])
    .then(function(){ applyingStruct = false; });
}
function applyDark(on){
  darkOn = !!on;
  document.body.classList.toggle("dark", darkOn);
  Plotly.relayout(gd, darkOn
    ? {paper_bgcolor:"#1e1e1e", plot_bgcolor:"#1e1e1e", "font.color":"#e6e6e6"}
    : {paper_bgcolor:"white", plot_bgcolor:"white", "font.color":"#2a3f5f"}
  ).then(applyStruct);
}

var embedded = (window.self !== window.top);
if (!embedded){
  var dm = document.createElement("div");
  dm.style.cssText = "position:fixed;top:10px;right:14px;z-index:1000;font-family:sans-serif;font-size:13px;display:flex;align-items:center;gap:8px";
  dm.innerHTML = "<span>Dark mode</span><label class='switch'><input type='checkbox' id='dm-toggle'><span class='sw'></span></label>";
  document.body.appendChild(dm);
  document.getElementById("dm-toggle").addEventListener("change", function(e){ applyDark(e.target.checked); });
}
window.addEventListener("message", function(ev){
  if (ev.data && ev.data.type === "td-dark"){ applyDark(ev.data.on); }
});

function esc(s){ return (s+"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

// Colour key pinned to the left edge; its content tracks the active colour-mode
// button. The mode list is derived from MODE_DEFS so it adapts when a scope
// drops a mode (e.g. impact, when there are no citations).
var MODES = MODE_DEFS.map(function(m){ return m.key; });
var LEG_TAB = 24;
var leg = document.createElement("div");
leg.id = "leg";
leg.style.cssText = "position:fixed;right:0;top:16px;z-index:900;font-family:sans-serif;font-size:12px;"
  + "display:flex;align-items:stretch;transition:transform .2s ease";
leg.innerHTML =
    "<div id='leg-handle' style='width:" + LEG_TAB + "px;display:flex;flex-direction:column;align-items:center;"
  + "justify-content:center;gap:8px;background:#f4f5f7;border:1px solid #ccc;border-right:none;"
  + "border-radius:8px 0 0 8px;cursor:pointer;user-select:none'>"
  + "<span id='leg-caret'>▶</span><span style='writing-mode:vertical-rl;letter-spacing:1px;color:#999;font-size:10px'>KEY</span></div>"
  + "<div id='leg-panel' style='width:248px;background:#fff;border:1px solid #ccc;border-right:none;"
  + "box-shadow:0 2px 10px rgba(0,0,0,.15);padding:9px 11px'>"
  + "<b id='leg-title' style='display:block;margin-bottom:6px'>Key</b><div id='leg-body'></div></div>";
document.body.appendChild(leg);
var legOpen = true;
document.getElementById("leg-handle").addEventListener("click", function(){
  legOpen = !legOpen;
  leg.style.transform = legOpen ? "translateX(0)" : "translateX(calc(100% - " + LEG_TAB + "px))";
  document.getElementById("leg-caret").textContent = legOpen ? "▶" : "◀";
});

function legBar(stops){
  return "<div style='height:13px;border-radius:3px;border:1px solid rgba(128,128,128,.5);"
       + "background:linear-gradient(to right," + stops.join(",") + ")'></div>";
}
function legSwatch(c, label){
  return "<div style='display:flex;align-items:center;gap:7px;margin:4px 0'><span style='display:inline-block;"
       + "width:14px;height:14px;border-radius:3px;background:" + c + ";border:1px solid rgba(128,128,128,.55)'></span>" + esc(label) + "</div>";
}
function renderLegend(idx){
  var mode = MODES[idx] || "identity";
  var L = LEGEND[mode];
  var title = document.getElementById("leg-title");
  var body = document.getElementById("leg-body");
  if (mode === "identity"){
    title.textContent = "Key — Theme";
    body.innerHTML = "<div style='color:#666'>" + esc(L.note) + "</div>";
  } else if (mode === "impact"){
    title.textContent = "Key — Impact";
    var ticks = "<div style='position:relative;height:15px;margin-top:3px'>";
    L.ticks.forEach(function(t){
      ticks += "<span style='position:absolute;left:" + t[1] + "%;transform:translateX(-50%);color:#666'>" + esc(t[0]) + "</span>";
    });
    ticks += "</div>";
    body.innerHTML = "<div style='margin-bottom:5px;color:#444'>" + esc(L.title) + "</div>" + legBar(L.stops) + ticks;
  } else if (mode === "growth"){
    title.textContent = "Key — Growth";
    var lab = "<div style='display:flex;justify-content:space-between;color:#666;margin-top:3px'><span>"
            + esc(L.left) + "</span><span>" + esc(L.mid) + "</span><span>" + esc(L.right) + "</span></div>";
    body.innerHTML = "<div style='margin-bottom:5px;color:#444'>" + esc(L.title) + "</div>" + legBar(L.stops) + lab
                   + "<div style='margin-top:7px'>" + L.extra.map(function(e){ return legSwatch(e[0], e[1]); }).join("") + "</div>";
  } else {
    title.textContent = "Key — Emerging";
    body.innerHTML = "<div style='margin-bottom:5px;color:#444'>" + esc(L.title) + "</div>"
                   + L.swatches.map(function(s){ return legSwatch(s[0], s[1]); }).join("");
  }
}
var modebar = document.createElement("div");
modebar.id = "modebar";
MODE_DEFS.forEach(function(m, i){
  var b = document.createElement("button");
  b.className = "modebtn"; b.type = "button"; b.textContent = m.label; b.dataset.idx = i;
  b.addEventListener("click", function(){ setMode(i); });
  modebar.appendChild(b);
});
gd.parentNode.insertBefore(modebar, gd);

var curMode = 0;
function setMode(i){
  curMode = i;
  var m = MODE_DEFS[i];
  Plotly.restyle(gd, {"marker.colors": [m.bg], "textfont.color": [m.fg]}, [0]).then(applyStruct);
  modebar.querySelectorAll(".modebtn").forEach(function(el, j){ el.classList.toggle("active", j === i); });
  renderLegend(i);
}
setMode(0);

function card(val, label, tip){
  return "<div class='statcard'>"
       + "<div style='font-size:1.35em;font-weight:700;color:#222'>" + val + "</div>"
       + "<div style='font-size:0.72em;color:#777;text-transform:uppercase;letter-spacing:.04em'>" + label + "</div>"
       + "<span class='tip'>" + esc(tip) + "</span></div>";
}
function cardRow(cards){ return "<div style='display:flex;flex-wrap:wrap;gap:8px;margin:12px 0'>" + cards + "</div>"; }
function names(arr, k){ return arr.slice(0, k || 3).map(function(t){ return esc(t[0]); }).join(", "); }
function fmt(n){ return (n + "").replace(/\\B(?=(\\d{3})+(?!\\d))/g, ","); }
function prevDecade(d){ return (parseInt(d, 10) - 10) + "s"; }
function listBlock(label, items){
  return "<p style='color:#555;margin:7px 0'><b>" + esc(label) + ":</b><br>"
       + items.map(function(s){ return "&nbsp;&nbsp;" + s; }).join("<br>") + "</p>";
}

function renderRoot(){
  var m = META.root;
  var cards = card(fmt(m.n_papers), "papers", "Total research papers in the corpus.")
         + card(m.n_topics, "topics", "Distinct topics discovered (excluding the noise/outlier cluster).")
         + card(m.ymin + "–" + m.ymax, "years", "Range of publication years covered.")
         + card(m.n_decades, "decades", "Number of decade buckets.");
  if (m.total_citations != null){
    cards += card(fmt(m.total_citations), "total cites", "Total citations across every paper in the corpus.")
           + card(m.median_citations, "median cites", "Median citations of a paper across the whole corpus.");
  }
  var html = "<h2 style='margin-bottom:4px'>" + esc(m.title) + "</h2>"
       + "<p style='color:#555;max-width:860px'>" + esc(m.blurb) + "</p>"
       + cardRow(cards)
       + "<p style='color:#555;margin:7px 0'>Across " + m.n_decades + " decades the most prevalent topics have been "
       + "<b>" + names(m.top_topics, 3) + "</b>, and the busiest decade was <b>" + esc(m.busiest_decade) + "</b>.</p>"
       + "<p style='color:#555;margin:7px 0'><b>Most prevalent topics overall:</b> "
       + m.top_topics.map(function(t){ return esc(t[0]) + " (" + fmt(t[1]) + ")"; }).join(", ") + "</p>";
  if (m.top_cited_topic){
    html += "<p style='color:#555;margin:7px 0'><b>Highest-impact topic:</b> " + esc(m.top_cited_topic[0])
          + " (" + fmt(m.top_cited_topic[1]) + " total citations)</p>";
  }
  if (m.flagship){
    var f = m.flagship;
    var flink = f.u ? "<a href='" + esc(f.u) + "' target='_blank' rel='noopener'>" + esc(f.t) + "</a>" : esc(f.t);
    html += "<p style='color:#555;margin:7px 0'><b>Most cited paper:</b> " + flink
          + " <span style='color:#999'>(" + f.y + ", " + fmt(f.c) + " citations)</span>"
          + (f.a ? "<br><span style='color:#777;font-size:0.9em'>" + esc(f.a) + "</span>" : "") + "</p>";
  }
  html += "<p style='color:#999;font-size:.9em'>Click a decade for its era summary, or a topic tile for full stats and papers.</p>";
  return html;
}

function renderDecade(decade){
  var m = META.decades[decade];
  if (!m) { return "<p style='color:#777'>No data for " + esc(decade) + ".</p>"; }
  var sub = fmt(m.n_papers) + " papers · " + m.n_topics + " topics · " + m.pct_corpus + "% of the corpus";
  if (m.median_cites != null) { sub += " · median " + m.median_cites + " citations"; }
  var html = "<h2 style='margin-bottom:2px'>" + esc(decade) + "</h2>"
           + "<p style='color:#777;margin-top:0'>" + sub + "</p>"
           + "<p style='color:#555;max-width:860px'>In the " + esc(decade) + ", the largest research areas were "
           + "<b>" + names(m.top_topics, 3) + "</b>.</p>";
  html += listBlock("Biggest areas", m.top_topics.map(function(t){
            return esc(t[0]) + " — " + fmt(t[1]) + " papers (" + t[2] + "%)"; }));
  if (m.rising || m.falling){
    var rf = [];
    if (m.rising) rf.push("<b>Rising fastest:</b> " + esc(m.rising[0]) + " (+" + m.rising[1] + "% share)");
    if (m.falling) rf.push("<b>Falling fastest:</b> " + esc(m.falling[0]) + " (" + m.falling[1] + "% share)");
    html += "<p style='color:#555;margin:7px 0'>" + rf.join(" &nbsp;·&nbsp; ") + "</p>";
  }
  if (m.emerging && m.emerging.length){
    html += "<p style='color:#555;margin:7px 0'><b>New this decade:</b> " + m.emerging.map(esc).join(", ") + "</p>";
  }
  if (m.faded && m.faded.length){
    html += "<p style='color:#555;margin:7px 0'><b>Faded from " + esc(prevDecade(decade)) + ":</b> "
          + m.faded.map(esc).join(", ") + "</p>";
  }
  if (m.top_cited){
    html += listBlock("Highest impact (total citations)", m.top_cited.map(function(t){
              return esc(t[0]) + " — " + fmt(t[1]) + " citations"; }));
  }
  html += "<p style='color:#999;font-size:.9em'>Click a topic tile within this decade for its full stats and papers.</p>";
  return html;
}

function renderTheme(decade, theme){
  var m = THEMES[decade + "||" + theme];
  if (!m) { return "<p style='color:#777'>No data for " + esc(theme) + ".</p>"; }
  var html = "<h2 style='margin-bottom:2px'>" + esc(m.name) + "</h2>"
           + "<p style='color:#777;margin-top:0'>" + esc(decade) + " · " + fmt(m.n_papers)
           + " papers · " + m.n_topics + " topics · " + m.pct_decade + "% of the decade</p>";
  html += listBlock("Topics in this theme", m.topics.map(function(t){
            return esc(t[0]) + " — " + fmt(t[1]) + " papers"; }));
  html += "<p style='color:#999;font-size:.9em'>Click a topic tile within this theme for its full stats and papers.</p>";
  return html;
}

function renderTopic(decade, tid){
  var entry = DATA[decade + "||" + tid];
  if (!entry) { return null; }
  var rows = entry.rows || [];
  var s = entry.stats || {};
  var growthTip = "Change in this topic's SHARE of the decade's papers versus the previous decade — share-based, so the corpus growing over time does not fake growth. 'new' = no papers in the previous decade.";
  var growthCard = "";
  var g = entry.growth;
  if (g){
    if (g.state === "up" || g.state === "down"){
      growthCard = card((g.pct >= 0 ? "↑ +" : "↓ ") + g.pct + "%", "growth vs " + g.prev, growthTip);
    } else if (g.state === "new"){
      growthCard = card("new", "vs " + g.prev, growthTip);
    } else {
      growthCard = card("—", "no prior decade", growthTip);
    }
  }
  var cards = card(s.n, "papers", "Number of papers assigned to this topic in this decade.");
  if (s.med != null){
    cards += card(s.med, "median cites", "Citations of the typical paper: half got more, half got fewer. Robust to a few blockbusters.")
           + card(s.mean, "mean cites", "Average citations per paper. Pulled up by a few highly-cited papers, so compare it against the median.")
           + card(s.tot, "total cites", "Sum of citations across all the topic papers: its overall citation footprint.")
           + card(s.h, "h-index", "h papers here each have at least h citations. A single impact measure that is not dominated by one outlier.")
           + card(s.uncpct + "%", "uncited", "Share of papers with zero recorded citations.");
  }
  cards += growthCard;
  var html = "<h2 style='margin-bottom:2px'>" + esc(entry.name) + "</h2>"
           + (entry.kw ? "<p style='color:#888;margin:0 0 4px;font-style:italic'>" + esc(entry.kw) + "</p>" : "")
           + "<p style='color:#777;margin-top:0'>" + esc(decade) + " · " + s.n + " papers · "
           + s.ymin + "–" + s.ymax + "</p>"
           + cardRow(cards);
  if (s.auth && s.auth.length){
    html += "<p style='color:#555;margin:6px 0'><b>Top authors:</b> "
          + s.auth.map(function(a){ return esc(a[0]) + " (" + a[1] + ")"; }).join(", ") + "</p>";
  }
  if (rows.length && rows[0].c != null){
    var f = rows[0];
    var flink = f.u ? "<a href='" + esc(f.u) + "' target='_blank' rel='noopener'>" + esc(f.t) + "</a>" : esc(f.t);
    html += "<p style='color:#555;margin:6px 0 14px'><b>Most cited:</b> " + flink
          + " <span style='color:#999'>(" + f.y + ", " + f.c + " citations)</span>"
          + (f.a ? "<br><span style='color:#777;font-size:0.9em'>" + esc(f.a) + "</span>" : "") + "</p>";
  }
  html += "<ol>";
  rows.forEach(function(r){
    var link = r.u ? "<a href='" + esc(r.u) + "' target='_blank' rel='noopener'>" + esc(r.t) + "</a>" : esc(r.t);
    var meta = " <span style='color:#999'>(" + r.y + (r.c != null ? ", " + r.c + " citations" : "") + ")</span>";
    var auth = r.a ? "<br><span style='color:#555;font-size:0.9em'>" + esc(r.a) + "</span>" : "";
    html += "<li style='margin-bottom:8px'>" + link + meta + auth + "</li>";
  });
  html += "</ol>";
  if (entry.more){
    html += "<p style='color:#999;font-size:.9em'>… and " + fmt(entry.more)
          + " more papers (showing the most recent " + rows.length + ").</p>";
  }
  return html;
}

gd.on("plotly_click", function(d){
  var parts = (d.points[0].id || "").split("/");
  var html;
  if (parts.length <= 1) { html = renderRoot(); }
  else if (parts.length === 2) { html = renderDecade(parts[1]); }
  else {
    // root/decade/theme[/topic_id]: a topic leaf ends in its numeric id (and is
    // in DATA); anything else at depth ≥3 is a theme tile. (Theme names carry no
    // "/", so the decade is always parts[1] and the topic id the last part.)
    var dec = parts[1], last = parts[parts.length - 1];
    html = DATA[dec + "||" + last] ? renderTopic(dec, last)
                                   : renderTheme(dec, parts.slice(2).join("/"));
  }
  if (html === null) { return; }
  panel.innerHTML = html;
});
"""


def _growth_hover(state, pct, prev) -> str:
    """The growth line appended to a topic tile's hover."""
    if state == "na":
        return "<br>growth: — (no earlier decade)"
    if state == "new":
        return f"<br><b>new</b> vs {prev}"
    return f"<br>growth vs {prev}: {pct:+.0f}% of share"


def _nodes(agg, has_cites, root_label, group_colors):
    """Flatten the aggregate into parallel node arrays for go.Treemap.

    The hierarchy is root → decade → theme → topic: topics are grouped under
    their overarching theme, and the theme tile carries the theme's colour
    (constant across every colour mode — the modes recolour the topic leaves).

    Returns ids, labels, parents, values, hover-html, and the colour pairs —
    (bg, fg) for topic-identity mode, the growth-vs-previous-decade view, the
    emerging-topics view, and (when citations exist) the median-citation heatmap.

    Decades are emitted most-recent-first (2020s leftmost); themes and topics are
    pre-sorted largest-first; the trace is built with sort=False so Plotly keeps
    that order instead of re-sorting every sector by size.
    """
    decades = sorted(agg["decade"].unique(), reverse=True)  # 2020s … left → right

    ids, labels, parents, values, hovers = [], [], [], [], []
    bg_id, fg_id = [], []
    bg_imp, fg_imp = [], []
    bg_grow, fg_grow, bg_new, fg_new = [], [], [], []

    if has_cites:
        # Median citations are heavily right-skewed, so normalise on a log scale —
        # that spreads the typical tiles across the palette.
        med = agg["median_citations"]
        log_lo = math.log1p(med.min())
        log_span = (math.log1p(med.max()) - log_lo) or 1.0

    def push(node_id, label, parent, value, hover_html, ident, imp, grow, new):
        ids.append(node_id); labels.append(label.replace(" · ", "<br>"))
        parents.append(parent); values.append(value); hovers.append(hover_html)
        bg_id.append(ident[0]); fg_id.append(ident[1])
        bg_grow.append(grow[0]); fg_grow.append(grow[1])
        bg_new.append(new[0]); fg_new.append(new[1])
        if has_cites:
            bg_imp.append(imp[0]); fg_imp.append(imp[1])

    neutral = (NEUTRAL_BG, NEUTRAL_FG)
    total = int(agg["papers"].sum())
    push(root_label, root_label, "", total, f"<b>{root_label}</b><br>{total} papers",
         neutral, neutral, neutral, neutral)

    for decade in decades:
        dsub = agg[agg["decade"] == decade]
        dec_papers = int(dsub["papers"].sum())
        dec_id = f"{root_label}/{decade}"
        push(dec_id, decade, root_label, dec_papers, f"<b>{decade}</b><br>{dec_papers} papers",
             neutral, neutral, neutral, neutral)

        # Themes ordered largest-first within the decade, topics largest-first
        # within each theme.
        theme_papers = dsub.groupby("theme")["papers"].sum().sort_values(ascending=False)
        for theme in theme_papers.index:
            tsub = dsub[dsub["theme"] == theme].sort_values("papers", ascending=False)
            tp = int(tsub["papers"].sum())
            theme_id = f"{dec_id}/{theme}"
            gcol = group_colors.get(theme, NEUTRAL_BG)
            gpair = (gcol, _text_on(gcol))
            thover = f"<b>{theme}</b><br>{tp} papers · {len(tsub)} topics"
            # Theme tiles keep their group colour in every mode (only the topic
            # leaves recolour), so the same pair is passed for all four.
            push(theme_id, theme, dec_id, tp, thover, gpair, gpair, gpair, gpair)

            for r in tsub.itertuples():
                # Default ("Theme") mode: each subtopic takes its overarching
                # theme's colour, so a theme reads as one coloured block.
                grow = _growth_color(r.growth_state, r.l2fc)
                is_new = r.growth_state == "new"
                new_bg = GROWTH_NEW if is_new else EMERGE_GREY
                hover = (f"<b>{r.topic}</b><br><i>{r.keywords}</i>"
                         f"<br>{theme}<br>{r.papers} papers")
                imp = neutral
                if has_cites:
                    imp_c = sample_colorscale("Turbo", (math.log1p(r.median_citations) - log_lo) / log_span)[0]
                    imp = (imp_c, _text_on(imp_c))
                    hover += (f"<br>median {r.median_citations:.0f} · mean {r.mean_citations:.1f} citations"
                              f"<br>total {int(r.total_citations)} citations")
                hover += _growth_hover(r.growth_state, r.growth_pct, r.prev_decade)
                push(f"{theme_id}/{r.topic_id}", r.topic, theme_id, int(r.papers), hover,
                     gpair, imp, (grow, _text_on(grow)),
                     (new_bg, _text_on(new_bg)))

    struct = [i for i, c in enumerate(bg_id) if c == NEUTRAL_BG]
    return dict(ids=ids, labels=labels, parents=parents, values=values, hovers=hovers,
                bg_id=bg_id, fg_id=fg_id, bg_imp=bg_imp, fg_imp=fg_imp,
                bg_grow=bg_grow, fg_grow=fg_grow, bg_new=bg_new, fg_new=fg_new,
                struct=struct)


def plot(scope, agg, papers, meta, group_colors, has_cites):
    root_label = SCOPE_TITLES.get(scope, scope)
    n = _nodes(agg, has_cites, root_label, group_colors)

    fig = go.Figure(go.Treemap(
        ids=n["ids"], labels=n["labels"], parents=n["parents"], values=n["values"],
        branchvalues="total",
        texttemplate="%{label}", textposition="middle center",
        customdata=n["hovers"], hovertemplate="%{customdata}<extra></extra>",
        marker=dict(colors=n["bg_id"], line=dict(width=1, color="white")),
        textfont=dict(color=n["fg_id"]),
        tiling=dict(pad=2),
        sort=False,
    ))

    # Colour modes are driven by custom HTML buttons (built in the post-script),
    # not Plotly's updatemenus. Impact only appears when the scope has citations.
    modes = [
        {"label": "Theme", "key": "identity", "bg": n["bg_id"], "fg": n["fg_id"]},
    ]
    if has_cites:
        modes.append({"label": "Impact (median citations)", "key": "impact",
                      "bg": n["bg_imp"], "fg": n["fg_imp"]})
    modes += [
        {"label": "Growth vs previous decade", "key": "growth", "bg": n["bg_grow"], "fg": n["fg_grow"]},
        {"label": "Emerging (new topics)", "key": "emerging", "bg": n["bg_new"], "fg": n["fg_new"]},
    ]
    fig.update_layout(
        margin=dict(t=34, l=10, r=10, b=10), height=800, template="plotly_white",
    )

    legend = {
        "identity": {"note": "Each subtopic takes the colour of its overarching theme."},
        "growth": {"title": "Change in share vs the previous decade",
                   "stops": [sample_colorscale("RdYlGn", i / 8)[0] for i in range(9)],
                   "left": "−75% (shrank)", "mid": "flat", "right": "+300% (grew)",
                   "extra": [[GROWTH_NA, "new / no prior decade (no comparison)"]]},
        "emerging": {"title": "Topics new to their decade",
                     "swatches": [[GROWTH_NEW, "new this decade"], [EMERGE_GREY, "existing topic"]]},
    }
    if has_cites:
        med = agg["median_citations"]
        log_lo = math.log1p(med.min())
        log_span = (math.log1p(med.max()) - log_lo) or 1.0
        tick_vals = [v for v in (0, 10, 50, 150, int(med.max())) if v <= med.max()]
        legend["impact"] = {"title": "Median citations per paper (log scale)",
                            "stops": [sample_colorscale("Turbo", i / 8)[0] for i in range(9)],
                            "ticks": [[str(v), round((math.log1p(v) - log_lo) / log_span * 100)]
                                      for v in tick_vals]}

    data_json = json.dumps(papers, ensure_ascii=False).replace("</", "<\\/")
    meta_json = json.dumps(meta, ensure_ascii=False).replace("</", "<\\/")
    themes_json = json.dumps(meta.get("themes", {}), ensure_ascii=False).replace("</", "<\\/")
    legend_json = json.dumps(legend, ensure_ascii=False)
    modes_json = json.dumps(modes)
    struct_json = json.dumps(n["struct"])
    neutral_json = json.dumps({"lbg": NEUTRAL_BG, "lfg": NEUTRAL_FG,
                               "dbg": DARK_NEUTRAL_BG, "dfg": DARK_NEUTRAL_FG})
    post = (_POST_SCRIPT.replace("__DATA__", data_json).replace("__META__", meta_json)
            .replace("__THEMES__", themes_json)
            .replace("__LEGEND__", legend_json).replace("__MODES__", modes_json)
            .replace("__STRUCT__", struct_json).replace("__NEUTRAL__", neutral_json))
    dest = FIGURES_DIR / f"{NAME}_{scope}.html"
    fig.write_html(str(dest), include_plotlyjs="cdn", post_script=post)
    print(f"  wrote {dest.name}")


def main():
    for scope in SCOPE_TITLES:
        agg, papers, meta, group_colors, has_cites = build(scope)
        print(f"{SCOPE_TITLES[scope]}: {len(agg)} decade×theme×topic tiles, "
              f"{len(meta['themes'])} theme tiles, "
              f"{sum(len(v['rows']) for v in papers.values())} papers indexed, "
              f"{len(meta['decades'])} decade descriptions"
              f"{'' if has_cites else ' (no citation data — reduced)'}")
        plot(scope, agg, papers, meta, group_colors, has_cites)


if __name__ == "__main__":
    main()
