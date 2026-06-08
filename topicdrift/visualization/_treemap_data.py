"""
_treemap_data.py — Data transformation pipeline for the topic treemap.

Loads per-scope paper tables, maps topic labels and theme groups, computes
decade-level aggregates and growth metrics, and assembles the panel index
(per topic×decade stats + paper list) consumed by the layout layer.
"""

import math
import re

import pandas as pd

from topicdrift.constants import OUTLIER_TOPIC_ID
from topicdrift.visualization._common import (
    INTERIM_DIR,
    PROCESSED_DIR,
    SCOPE_TITLES,
    load_conf_paper_topics,
    scope_filter,
    short_label,
)

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

# For the conf scopes (no citations to rank by) the per topic×decade paper list
# is capped to the most recent CAP papers to keep the embedded JSON — and the
# page — a sane size; a "+N more" note flags the truncation.
CONF_LIST_CAP = 200

# |Δshare| below this counts as "flat" rather than "up" or "down" growth.
GROWTH_FLAT_PCT = 5.0


def _conf_titles() -> dict[int, dict[str, str]]:
    """{topic_id: {'title', 'keywords'}} for the global conf fit."""
    topics = pd.read_parquet(PROCESSED_DIR / "conf_topics.parquet")
    out: dict[int, dict[str, str]] = {}
    for _, r in topics.iterrows():
        tid = int(r["topic_id"])
        if tid == OUTLIER_TOPIC_ID:
            continue
        kw = short_label(r["top_words"])
        llm = str(r["llm_label"]).strip() if pd.notna(r["llm_label"]) else ""
        out[tid] = {"title": llm or kw, "keywords": kw}
    return out


def _topic_groups(scope: str) -> tuple[dict[int, str], dict[str, str]]:
    """({topic_id: theme}, {theme: colour}) for the scope's fit."""
    t = pd.read_parquet(PROCESSED_DIR / "conf_topics.parquet")
    tg = {int(r.topic_id): str(r.group) for r in t.itertuples() if pd.notna(r.group)}
    gc = {
        str(r.group): str(r.group_color)
        for r in t.itertuples()
        if pd.notna(r.group) and pd.notna(r.group_color)
    }
    return tg, gc


# Trailing DBLP disambiguation suffix on a name, e.g. "Xu Liu 0001" → "Xu Liu".
_DBLP_SUFFIX = re.compile(r"\s+\d{4}$")
# Authors shown before collapsing the rest into "et al." (keeps the subline short).
MAX_AUTHORS = 4


def _format_authors(names) -> str:
    """DBLP author array → display string, suffixes stripped, capped + 'et al.'."""
    try:
        seq = list(names) if names is not None else []
    except TypeError:
        return ""
    clean = [_DBLP_SUFFIX.sub("", str(n)).strip() for n in seq if str(n).strip()]
    if not clean:
        return ""
    if len(clean) > MAX_AUTHORS:
        return ", ".join(clean[:MAX_AUTHORS]) + ", et al."
    return ", ".join(clean)


def _paper_links(keys: set[str]) -> pd.DataFrame:
    """Per-paper external link + author line for the given papers, by dblp_key.

    These fields (`ee`, `url`, `doi`, `authors`) live in the DBLP slice, not in
    conf_enriched, so we read them here. The link prefers DBLP's `ee` (a
    resolvable https link), falls back to a DOI URL, then the DBLP record page."""
    src = pd.read_parquet(
        INTERIM_DIR / "dblp_conf.parquet",
        columns=["dblp_key", "ee", "url", "doi", "authors"],
    )
    src = src[src["dblp_key"].isin(keys)]
    rows = []
    for r in src.itertuples():
        ee = r.ee.strip() if isinstance(r.ee, str) else ""
        doi = r.doi.strip() if isinstance(r.doi, str) else ""
        url = r.url.strip() if isinstance(r.url, str) else ""
        if ee:
            link = ee
        elif doi:
            link = "https://doi.org/" + doi
        elif url:
            link = url if url.startswith("http") else "https://dblp.org/" + url.lstrip("/")
        else:
            link = ""
        rows.append((r.dblp_key, link, _format_authors(r.authors)))
    return pd.DataFrame(rows, columns=["dblp_key", "ee", "authors_str"])


def scope_source(scope: str) -> dict:
    """Bundle the per-paper table, titles, root label and blurb for a scope."""
    root_label = SCOPE_TITLES.get(scope, scope)
    blurb = ROOT_BLURBS.get(scope, "")
    pt = scope_filter(load_conf_paper_topics(), scope)
    titles = pd.read_parquet(INTERIM_DIR / "conf_enriched.parquet", columns=["dblp_key", "title"])
    pt = pt.merge(titles, on="dblp_key", how="left")
    # Attach the EE link (clickable title) + author subline for each paper. Only
    # the rows that actually get listed (capped per topic×decade in build())
    # reach the HTML, so the embedded JSON stays bounded.
    pt = pt.merge(_paper_links(set(pt["dblp_key"])), on="dblp_key", how="left")
    return {
        "pt": pt,
        "titles": _conf_titles(),
        "root_label": root_label,
        "blurb": blurb,
    }


def _prev_decade(decade: str) -> str:
    """'2010s' → '2000s'."""
    return f"{int(decade[:-1]) - 10}s"


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
        if pdec not in have:
            states.append("na")
            pcts.append(float("nan"))
            l2s.append(float("nan"))
        elif prev <= 0.0:
            states.append("new")
            pcts.append(float("nan"))
            l2s.append(float("nan"))
        else:
            pct = (r.share / prev - 1.0) * 100.0
            states.append(
                "up" if pct > GROWTH_FLAT_PCT else "down" if pct < -GROWTH_FLAT_PCT else "flat"
            )
            pcts.append(pct)
            l2s.append(math.log2(r.share / prev))
    agg["prev_decade"] = prevs
    agg["growth_state"] = states
    agg["growth_pct"] = pcts
    agg["l2fc"] = l2s
    return agg


def _stats(grp) -> dict:
    """Per topic×decade headline stats for the click panel."""
    return {"n": len(grp), "ymin": int(grp["year"].min()), "ymax": int(grp["year"].max())}


def _meta(agg, pt, root_label, blurb) -> dict:
    """Descriptions for the structural tiles (root and each decade)."""
    corpus_total = int(agg["papers"].sum())

    by_topic = (
        agg.groupby(["topic_id", "topic"], as_index=False)["papers"].sum().nlargest(6, "papers")
    )
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

    members = {
        d: dict(zip(s["topic_id"].astype(int), s["topic"])) for d, s in agg.groupby("decade")
    }
    sizes = {
        (d, int(r.topic_id)): int(r.papers)
        for d, s in agg.groupby("decade")
        for r in s.itertuples()
    }

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

        decades[decade] = {
            "n_papers": total,
            "n_topics": int(len(sub)),
            "pct_corpus": round(total / corpus_total * 100, 1),
            "top_topics": [
                [r.topic, int(r.papers), round(r.papers / total * 100)]
                for r in prevalent.itertuples()
            ],
            "emerging": [r.topic for r in emerging.itertuples()],
            "faded": [members[pdec][t] for t in faded],
            "rising": [up.iloc[0]["topic"], round(up.iloc[0]["growth_pct"])] if len(up) else None,
            "falling": [down.iloc[0]["topic"], round(down.iloc[0]["growth_pct"])]
            if len(down)
            else None,
        }
    return {"root": root, "decades": decades}


def _theme_meta(agg) -> dict:
    """Per decade×theme summary for the click panel when a theme tile is clicked."""
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
    """Return (tile aggregate, panel index, meta, group_colors).

    panel index: {'<decade>||<topic_id>': {name, stats, rows}} for topic tiles.
    meta: {'root', 'decades', 'themes'} — the structural-tile descriptions."""
    src = scope_source(scope)
    pt, titles = src["pt"], src["titles"]
    groups, group_colors = _topic_groups(scope)
    pt = pt[pt["topic_id"].isin(titles)].copy()
    pt["decade"] = (pt["year"] // 10 * 10).astype(int).astype(str) + "s"
    pt["topic"] = pt["topic_id"].map(lambda t: titles[t]["title"])
    pt["keywords"] = pt["topic_id"].map(lambda t: titles[t]["keywords"])
    pt["theme"] = pt["topic_id"].map(groups).fillna("Other")

    agg = (
        pt.groupby(["decade", "theme", "topic_id", "topic", "keywords"])
        .agg(papers=("dblp_key", "size"))
        .reset_index()
    )
    agg = _add_growth(agg)

    growth = {
        (r.decade, int(r.topic_id)): {
            "state": r.growth_state,
            "prev": r.prev_decade,
            "pct": None if math.isnan(r.growth_pct) else round(r.growth_pct),
        }
        for r in agg.itertuples()
    }

    papers: dict[str, dict] = {}
    for (decade, tid), grp in pt.sort_values("year", ascending=False).groupby(
        ["decade", "topic_id"]
    ):
        listed = grp.head(CONF_LIST_CAP)
        rows = []
        for r in listed.itertuples():
            row = {"t": r.title or "(untitled)", "y": int(r.year)}
            ee = getattr(r, "ee", None)
            if isinstance(ee, str) and ee:
                row["u"] = ee  # renders as a clickable link in _treemap_layout
            authors = getattr(r, "authors_str", None)
            if isinstance(authors, str) and authors:
                row["a"] = authors  # renders as an author subline in _treemap_layout
            rows.append(row)
        entry = {
            "name": titles[tid]["title"],
            "kw": titles[tid]["keywords"],
            "stats": _stats(grp),
            "growth": growth.get((decade, int(tid))),
            "rows": rows,
        }
        if len(grp) > CONF_LIST_CAP:
            entry["more"] = int(len(grp) - CONF_LIST_CAP)
        papers[f"{decade}||{tid}"] = entry
    meta = _meta(agg, pt, src["root_label"], src["blurb"])
    meta["themes"] = _theme_meta(agg)
    return agg, papers, meta, group_colors
