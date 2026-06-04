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
import logging

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

log = logging.getLogger(__name__)

NAME = "topic_group_streamgraph"
DEFAULT_COLOR = "#94a3b8"


def build_scope_data(pt: pd.DataFrame, id_to_label: dict, group_order: list[str]) -> dict:
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
    tb = df.groupby(["group", "bucket", "llm_label"]).size().rename("freq").reset_index()
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
                # Native hover is disabled (hovermode=False below); a custom
                # cursor-driven tooltip in the post-script handles band hover.
                hoverinfo="skip",
                fillcolor=c,
                opacity=0.8,
                line=dict(color=c, width=1.2),
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
        # Disable Plotly's built-in hover entirely — filled areas only surface
        # the trace name (not period/share), and "x unified" stacked all themes
        # into one flickering box. The post-script renders a custom tooltip.
        hovermode=False,
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
var dark = false;
var current = null;  // {group,bucket} of the open drilldown, or null for the hint

function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function rgba(hex,a){var r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return"rgba("+r+","+g+","+b+","+a+")";}

// Light/dark palettes for the custom (non-Plotly) UI — the toggle bar and the
// drilldown cards — so they track the page theme the figure is embedded in.
var PAL={
  light:{barBg:"#f8fafc",barBd:"#e2e8f0",lab:"#64748b",aBg:"#2563eb",aFg:"#fff",iBg:"#fff",iFg:"#374151",
         hint:"#94a3b8",cardBg:"#f8fafc",cardBd:"#e2e8f0",head:"#1e293b",sub:"#64748b",track:"#e2e8f0"},
  dark:{barBg:"#2a2a2a",barBd:"#3a3a3a",lab:"#9aa3ad",aBg:"#4a90d9",aFg:"#fff",iBg:"#1e1e1e",iFg:"#cbd5e1",
        hint:"#9aa3ad",cardBg:"#262626",cardBd:"#3a3a3a",head:"#e6e6e6",sub:"#9aa3ad",track:"#3a3a3a"}};
function pal(){return dark?PAL.dark:PAL.light;}

function traces(m){return DATA.groups.map(function(g){var c=COLORS[g]||"#94a3b8";return{
  type:"scatter",mode:"lines",stackgroup:"one",hoverinfo:"skip",
  x:DATA.buckets,y:DATA[m][g],name:g,fillcolor:rgba(c,0.8),line:{color:c,width:1.2}};});}
function layout(m){var y=m==="share"?{title:"Share of papers in period",tickformat:".0%"}:{title:"Papers in period",tickformat:""};
  return Object.assign({},base,{yaxis:Object.assign({},base.yaxis,y)});}
// Merge dark overrides (page/plot background, font, axes) onto a layout. Light
// mode returns the layout untouched so the template's own colours show through.
function themeLayout(l){
  if(!dark) return l;
  return Object.assign({},l,{paper_bgcolor:"#1e1e1e",plot_bgcolor:"#1e1e1e",
    font:Object.assign({},l.font||{},{color:"#e6e6e6"}),
    xaxis:Object.assign({},l.xaxis||{},{gridcolor:"#333",linecolor:"#555",zerolinecolor:"#333"}),
    yaxis:Object.assign({},l.yaxis||{},{gridcolor:"#333",linecolor:"#555",zerolinecolor:"#333"})});}
function render(){Plotly.react(gd,traces(mode),themeLayout(layout(mode)));}

var bar=document.createElement("div");
bar.style.cssText="display:flex;align-items:center;gap:14px;padding:8px 14px;margin-bottom:8px;border:1px solid;border-radius:8px;font-family:-apple-system,sans-serif;font-size:13px;transition:background .2s,border-color .2s;";
var lab=document.createElement("span");lab.textContent="Y-axis";lab.style.cssText="font-weight:600;";
var grp=document.createElement("span");grp.style.cssText="display:inline-flex;border:1px solid;border-radius:6px;overflow:hidden;";
var buttons=[];
["Share %","Paper Count"].forEach(function(t,i){var k=i===0?"share":"freq";var b=document.createElement("button");
  b.textContent=t;b.dataset.mode=k;b.style.cssText="border:none;padding:4px 14px;font-size:13px;cursor:pointer;";
  b.onclick=function(){mode=k;styleBar();render();renderPanel();};grp.appendChild(b);buttons.push(b);});
bar.appendChild(lab);bar.appendChild(grp);gd.parentNode.insertBefore(bar,gd);

function styleBar(){var p=pal();
  bar.style.background=p.barBg;bar.style.borderColor=p.barBd;grp.style.borderColor=p.barBd;lab.style.color=p.lab;
  buttons.forEach(function(b){var a=b.dataset.mode===mode;b.style.background=a?p.aBg:p.iBg;b.style.color=a?p.aFg:p.iFg;});}

var panel=document.createElement("div");
panel.style.cssText="font-family:-apple-system,sans-serif;max-width:1100px;margin:14px auto 0;padding:0 4px;";
gd.parentNode.insertBefore(panel,gd.nextSibling);
function renderPanel(){if(current)showPanel(current.group,current.bucket);else hint();}
function hint(){current=null;var p=pal();panel.innerHTML="<p style='color:"+p.hint+";font-size:0.875rem;margin:0;'>Click any band to see its sub-topics and each one's share of that theme.</p>";}

function showPanel(group,bucket){
  current={group:group,bucket:bucket};
  var e=(DATA.topics[group]||{})[String(bucket)]||[];var p=pal();
  if(!e.length){panel.innerHTML="<p style='color:"+p.hint+";'>No data.</p>";return;}
  var c=COLORS[group]||"#94a3b8",tot=e.reduce(function(s,t){return s+t.papers;},0);
  var h="<div style='border-top:3px solid "+c+";padding-top:12px;'><h3 style='margin:0 0 2px;font-size:1rem;color:"+p.head+";'>"+esc(group)+"</h3>"
    +"<p style='margin:0 0 12px;color:"+p.sub+";font-size:0.8rem;'>"+bucket+"–"+(bucket+4)+" \xb7 "+e.length+" sub-topics \xb7 "+tot+" papers \xb7 each bar = share of this theme</p>"
    +"<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px;'>";
  e.forEach(function(t){var pc=(t.share*100).toFixed(1);
    h+="<div style='background:"+p.cardBg+";border:1px solid "+p.cardBd+";border-radius:6px;padding:10px 12px;'>"
      +"<div style='font-weight:600;font-size:0.85rem;color:"+p.head+";line-height:1.3;'>"+esc(t.name)+"</div>"
      +"<div style='height:6px;border-radius:3px;background:"+p.track+";margin:6px 0 4px;overflow:hidden;'><div style='height:100%;width:"+pc+"%;background:"+c+";'></div></div>"
      +"<div style='font-size:0.78rem;color:"+p.sub+";'><b style='color:"+p.head+";'>"+pc+"%</b> of theme \xb7 "+t.papers+" papers</div></div>";});
  h+="</div></div>";panel.innerHTML=h;panel.scrollIntoView({behavior:"smooth",block:"nearest"});
}
// ── custom cursor-driven tooltip + drilldown ────────────────────────────────
// Plotly's native hover can't label a stacked band between its data points
// (filled-area hover only shows the trace name), so we map the cursor to a
// (theme, bucket) ourselves from the live axis ranges and drive both the
// floating tooltip and the click drilldown from the same lookup — which also
// guarantees clicks snap to a real bucket instead of an interpolated x.
var tip=document.createElement("div");
tip.style.cssText="position:fixed;pointer-events:none;z-index:9999;display:none;padding:6px 10px;border-radius:6px;border:1px solid;font-family:-apple-system,sans-serif;font-size:12px;line-height:1.35;box-shadow:0 2px 8px rgba(0,0,0,.18);white-space:nowrap;";
document.body.appendChild(tip);

// Pixel (relative to the plot div) → (theme, bucket) using the live axes.
function locate(evt){
  var fl=gd._fullLayout; if(!fl) return null;
  var xa=fl.xaxis, ya=fl.yaxis; if(!xa||!ya||!xa.range||!ya.range) return null;
  var bb=gd.getBoundingClientRect();
  var px=evt.clientX-bb.left-xa._offset, py=evt.clientY-bb.top-ya._offset;
  if(px<0||px>xa._length||py<0||py>ya._length) return null;
  var xr=xa.range, yr=ya.range;
  var xd=xr[0]+(px/xa._length)*(xr[1]-xr[0]);
  var yd=yr[1]-(py/ya._length)*(yr[1]-yr[0]);
  if(yd<0) return null;
  var bi=0,best=Infinity;                       // nearest 5-year bucket
  for(var i=0;i<DATA.buckets.length;i++){var d=Math.abs(DATA.buckets[i]-xd);if(d<best){best=d;bi=i;}}
  var cum=0,gi=-1;                              // which stacked band holds yd
  for(var j=0;j<DATA.groups.length;j++){cum+=DATA[mode][DATA.groups[j]][bi];if(yd<=cum){gi=j;break;}}
  if(gi<0) return null;                         // above the stack — empty space
  return {group:DATA.groups[gi],bucket:DATA.buckets[bi],value:DATA[mode][DATA.groups[gi]][bi]};
}

function showTip(evt,loc){var p=pal(),c=COLORS[loc.group]||"#94a3b8";
  var v=mode==="share"?(loc.value*100).toFixed(1)+"%":loc.value.toLocaleString()+" papers";
  var vlab=mode==="share"?"share":"papers";
  tip.style.background=dark?"#262626":"#fff";tip.style.borderColor=dark?"#3a3a3a":"#e2e8f0";tip.style.color=p.head;
  tip.innerHTML="<div style='font-weight:600;'><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:"+c+";margin-right:6px;vertical-align:middle;'></span>"+esc(loc.group)+"</div>"
    +"<div style='color:"+p.sub+";margin-top:2px;'>"+loc.bucket+"s \xb7 "+vlab+" <b style='color:"+p.head+";'>"+v+"</b></div>";
  tip.style.display="block";tip.style.left=(evt.clientX+14)+"px";tip.style.top=(evt.clientY+14)+"px";}
function hideTip(){tip.style.display="none";}

gd.addEventListener("mousemove",function(evt){var loc=locate(evt);if(loc)showTip(evt,loc);else hideTip();});
gd.addEventListener("mouseleave",hideTip);
gd.addEventListener("click",function(evt){var loc=locate(evt);if(loc)showPanel(loc.group,loc.bucket);});

// Dark mode is owned by the host page (the site posts {type:"td-dark",on} into
// the iframe on load and on toggle); repaint the figure + custom UI to match.
function applyDark(on){dark=!!on;document.body.style.background=dark?"#1e1e1e":"";styleBar();render();renderPanel();}
window.addEventListener("message",function(ev){if(ev.data&&ev.data.type==="td-dark"){applyDark(ev.data.on);}});

styleBar();hint();
})();
"""


def write_scope(scope: str, pt_scope: pd.DataFrame, id_to_label, group_color, group_order) -> None:
    data = build_scope_data(pt_scope, id_to_label, group_order)
    title = SCOPE_TITLES.get(scope, scope)
    fig = build_fig(data, group_color, title)
    post = _POST_SCRIPT.replace(
        "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    ).replace("__COLORS__", json.dumps(group_color, ensure_ascii=False))
    dest = FIGURES_DIR / f"{NAME}_{scope}.html"
    fig.write_html(str(dest), include_plotlyjs="cdn", post_script=post)
    log.info(
        "  wrote %s (%d themes, %d papers, %d buckets)",
        dest.name,
        len(data["groups"]),
        len(pt_scope),
        len(data["buckets"]),
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
