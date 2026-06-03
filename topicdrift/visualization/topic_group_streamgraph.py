"""
topic_group_streamgraph.py — Stacked-area theme drift, one figure per scope.

Each scope (ICSE / Top-10 / All) is a venue filter over the one global topic
space. For each, writes a stacked-area chart of theme share over 5-year periods
with a Share %/Paper Count toggle and click-to-drill-down showing a band's
sub-topics and their share of that theme (cards sum to 100%).

Writes: outputs/figures/topic_group_streamgraph_{scope}.html
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go

from topicdrift.visualization._common import (
    BUCKET_YEARS,
    FIGURES_DIR,
    SCOPE_TITLES,
    conf_group_registry,
    conf_topic_labels,
    load_conf_paper_topics,
    load_scopes,
    scope_filter,
)

NAME = "topic_group_streamgraph"
DEFAULT_COLOR = "#94a3b8"


def build_scope_data(
    pt: pd.DataFrame, id_to_label: dict, group_order: list[str]
) -> dict:
    """Per-scope payload: group time series + sub-topic drilldown (share-of-group)."""
    df = pt.dropna(subset=["group"]).copy()
    df["bucket"] = (df["year"] // BUCKET_YEARS) * BUCKET_YEARS
    df["llm_label"] = df["topic_id"].astype(int).map(id_to_label)

    buckets = sorted(int(b) for b in df["bucket"].unique())
    bucket_total = df.groupby("bucket").size()

    # group × bucket counts
    gb = df.groupby(["group", "bucket"]).size().rename("freq").reset_index()
    groups = [g for g in group_order if g in set(gb["group"])]

    share, freq = {}, {}
    for g in groups:
        sub = gb[gb["group"] == g].set_index("bucket")["freq"]
        freq[g] = [int(sub.get(b, 0)) for b in buckets]
        share[g] = [float(sub.get(b, 0)) / int(bucket_total.get(b, 1)) for b in buckets]

    # drilldown: sub-topic share OF ITS GROUP per bucket
    topics_detail: dict[str, dict[str, list[dict]]] = {}
    tb = (
        df.groupby(["group", "bucket", "llm_label"]).size().rename("freq").reset_index()
    )
    grp_bucket_total = gb.set_index(["group", "bucket"])["freq"].to_dict()
    for g in groups:
        topics_detail[g] = {}
        for b in buckets:
            rows = tb[(tb["group"] == g) & (tb["bucket"] == b)]
            denom = grp_bucket_total.get((g, b), 0) or 1
            entries = [
                {
                    "name": str(r["llm_label"]),
                    "papers": int(r["freq"]),
                    "share": round(int(r["freq"]) / denom, 4),
                }
                for _, r in rows.sort_values("freq", ascending=False).iterrows()
            ]
            if entries:
                topics_detail[g][str(b)] = entries

    return {
        "groups": groups,
        "buckets": buckets,
        "share": share,
        "freq": freq,
        "topics": topics_detail,
    }


# ── figure ────────────────────────────────────────────────────────────────--


def build_fig(data: dict, group_color: dict, title: str) -> go.Figure:
    fig = go.Figure()
    for g in data["groups"]:
        c = group_color.get(g, DEFAULT_COLOR)
        fig.add_trace(
            go.Scatter(
                x=data["buckets"],
                y=data["share"][g],
                name=g,
                mode="lines",
                stackgroup="one",
                hoveron="points+fills",
                fillcolor=c,
                opacity=0.8,
                line=dict(color=c, width=1.2),
                hovertemplate=f"<b>{g}</b><br>period: %{{x}}s<br>share: %{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Research theme evolution — {title}",
        xaxis=dict(
            title="5-year period",
            tickmode="array",
            tickvals=data["buckets"],
            ticktext=[f"{b}s" for b in data["buckets"]],
        ),
        yaxis=dict(title="Share of papers in period", tickformat=".0%"),
        legend=dict(title="Theme", x=1.01, y=1.0),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(t=60, l=60, r=240, b=60),
        height=560,
    )
    return fig


_POST_SCRIPT = r"""
(function () {
var DATA = __DATA__, COLORS = __COLORS__;
var gd = document.getElementById("{plot_id}");
var base = JSON.parse(JSON.stringify(gd.layout));
var mode = "share";

function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function rgba(hex,a){var r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return"rgba("+r+","+g+","+b+","+a+")";}

function traces(m){return DATA.groups.map(function(g){var c=COLORS[g]||"#94a3b8";return{
  type:"scatter",mode:"lines",stackgroup:"one",hoveron:"points+fills",
  x:DATA.buckets,y:DATA[m][g],name:g,fillcolor:rgba(c,0.8),line:{color:c,width:1.2},
  hovertemplate:"<b>"+esc(g)+"</b><br>period: %{x}s<br>"+(m==="share"?"share: %{y:.1%}":"papers: %{y}")+"<extra></extra>"};});}
function layout(m){var y=m==="share"?{title:"Share of papers in period",tickformat:".0%"}:{title:"Papers in period",tickformat:""};
  return Object.assign({},base,{yaxis:Object.assign({},base.yaxis,y)});}

var bar=document.createElement("div");
bar.style.cssText="display:flex;align-items:center;gap:14px;padding:8px 14px;margin-bottom:8px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-family:-apple-system,sans-serif;font-size:13px;";
var lab=document.createElement("span");lab.textContent="Y-axis";lab.style.cssText="color:#64748b;font-weight:600;";
var grp=document.createElement("span");grp.style.cssText="display:inline-flex;border:1px solid #cbd5e1;border-radius:6px;overflow:hidden;";
["Share %","Paper Count"].forEach(function(t,i){var k=i===0?"share":"freq";var b=document.createElement("button");
  b.textContent=t;b.dataset.mode=k;b.style.cssText="border:none;padding:4px 14px;font-size:13px;cursor:pointer;"+(i===0?"background:#2563eb;color:#fff;":"background:#fff;color:#374151;");
  b.onclick=function(){mode=k;grp.querySelectorAll("button").forEach(function(x){var a=x.dataset.mode===k;x.style.background=a?"#2563eb":"#fff";x.style.color=a?"#fff":"#374151";});
    Plotly.react(gd,traces(mode),layout(mode));hint();};grp.appendChild(b);});
bar.appendChild(lab);bar.appendChild(grp);gd.parentNode.insertBefore(bar,gd);

var panel=document.createElement("div");
panel.style.cssText="font-family:-apple-system,sans-serif;max-width:1100px;margin:14px auto 0;padding:0 4px;";
gd.parentNode.insertBefore(panel,gd.nextSibling);
function hint(){panel.innerHTML="<p style='color:#94a3b8;font-size:0.875rem;margin:0;'>Click any band to see its sub-topics and each one's share of that theme.</p>";}
hint();

function showPanel(group,bucket){
  var e=(DATA.topics[group]||{})[String(bucket)]||[];
  if(!e.length){panel.innerHTML="<p style='color:#94a3b8;'>No data.</p>";return;}
  var c=COLORS[group]||"#94a3b8",tot=e.reduce(function(s,t){return s+t.papers;},0);
  var h="<div style='border-top:3px solid "+c+";padding-top:12px;'><h3 style='margin:0 0 2px;font-size:1rem;color:#1e293b;'>"+esc(group)+"</h3>"
    +"<p style='margin:0 0 12px;color:#64748b;font-size:0.8rem;'>"+bucket+"–"+(bucket+4)+" \xb7 "+e.length+" sub-topics \xb7 "+tot+" papers \xb7 each bar = share of this theme</p>"
    +"<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px;'>";
  e.forEach(function(t){var p=(t.share*100).toFixed(1);
    h+="<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px 12px;'>"
      +"<div style='font-weight:600;font-size:0.85rem;color:#1e293b;line-height:1.3;'>"+esc(t.name)+"</div>"
      +"<div style='height:6px;border-radius:3px;background:#e2e8f0;margin:6px 0 4px;overflow:hidden;'><div style='height:100%;width:"+p+"%;background:"+c+";'></div></div>"
      +"<div style='font-size:0.78rem;color:#64748b;'><b style='color:#1e293b;'>"+p+"%</b> of theme \xb7 "+t.papers+" papers</div></div>";});
  h+="</div></div>";panel.innerHTML=h;panel.scrollIntoView({behavior:"smooth",block:"nearest"});
}
gd.on("plotly_click",function(ev){var pt=ev.points[0];showPanel(pt.data.name,pt.x);});
})();
"""


def write_scope(
    scope: str, pt_scope: pd.DataFrame, id_to_label, group_color, group_order
) -> None:
    data = build_scope_data(pt_scope, id_to_label, group_order)
    title = SCOPE_TITLES.get(scope, scope)
    fig = build_fig(data, group_color, title)
    post = _POST_SCRIPT.replace(
        "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    ).replace("__COLORS__", json.dumps(group_color, ensure_ascii=False))
    dest = FIGURES_DIR / f"{NAME}_{scope}.html"
    fig.write_html(str(dest), include_plotlyjs="cdn", post_script=post)
    print(
        f"  wrote {dest.name} ({len(data['groups'])} themes, "
        f"{len(pt_scope):,} papers, {len(data['buckets'])} buckets)"
    )


def main() -> None:
    id_to_label = conf_topic_labels()
    group_color, group_order = conf_group_registry()
    pt = load_conf_paper_topics()
    if "group" not in pt.columns:
        raise SystemExit(
            "conf_paper_topics has no `group` column — run "
            "apply_topic_groups.py --prefix conf_ first."
        )
    scopes = load_scopes()
    for scope in SCOPE_TITLES:
        pt_scope = scope_filter(pt, scope, scopes)
        if len(pt_scope):
            write_scope(scope, pt_scope, id_to_label, group_color, group_order)


if __name__ == "__main__":
    main()
