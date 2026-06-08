"""
Validation: topic coherence, assignment accuracy, assignment coverage, trend faithfulness.
Run: make validate  →  data/validation/*.json  outputs/figures/val_*.png
"""
from __future__ import annotations

import json, logging, re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from topicdrift.topic_model import lemmatizing_tokenizer, load_stopwords

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
INTER = ROOT / "data/interim"
OUT  = ROOT / "data/validation"
FIGS = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

log  = logging.getLogger(__name__)
SEED = 42

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": False, "legend.frameon": False,
})

_ABBREV = {
    "AI for Software Engineering":           "AI for SE",
    "Human Factors in Software Engineering": "Human Factors",
    "Requirements Engineering":              "Req. Eng.",
    "Defect Management":                     "Defect Mgmt",
    "Developer Tooling":                     "Dev Tooling",
    "Program Correctness":                   "Prog. Correct.",
}


def _savefig(fig, name: str) -> None:
    fig.savefig(FIGS / f"val_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _palette() -> dict[str, str]:
    tg = pd.read_parquet(PROC / "conf_topic_groups.parquet")
    return dict(zip(tg["group"], tg["color"]))


def _cooccur(docs, vocab):
    X = CountVectorizer(
        tokenizer=lemmatizing_tokenizer, token_pattern=None,
        vocabulary={w: i for i, w in enumerate(vocab)}, binary=True,
    ).fit_transform(docs)
    return np.asarray(X.sum(0), dtype=np.float64).flatten(), (X.T @ X).toarray().astype(np.float64)


def _npmi(words, D, df, co, w2i):
    idx = [w2i[w] for w in words if w in w2i]
    if len(idx) < 2:
        return float("nan")
    vals = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            dij = co[idx[i], idx[j]]
            if dij == 0:
                vals.append(-1.0)
                continue
            pi, pj, pij = df[idx[i]] / D, df[idx[j]] / D, dij / D
            vals.append((np.log2(pij) - np.log2(pi) - np.log2(pj)) / -np.log2(pij))
    return float(np.mean(vals))


def _umass(words, df, co, w2i):
    idx = [w2i[w] for w in words if w in w2i]
    if len(idx) < 2:
        return float("nan")
    return float(np.mean([
        np.log((co[idx[i], idx[j]] + 1) / max(df[idx[j]], 1))
        for i in range(1, len(idx)) for j in range(i)
    ]))


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, c - m), min(1.0, c + m)


def _gwet(y1, y2):
    a, b = np.array(y1, bool), np.array(y2, bool)
    p_o = (a == b).mean()
    pi  = (a.mean() + b.mean()) / 2
    p_e = 2 * pi * (1 - pi)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def _extract_theme(notes):
    if not isinstance(notes, str):
        return None
    m = re.search(r"[Ss]hould be\s+([^:;]+?)(?:\s+or\s+|\s*[:;]|$)", notes)
    if not m:
        return None
    raw = m.group(1).strip()
    for theme in pd.read_parquet(PROC / "conf_topic_groups.parquet")["group"]:
        if theme.lower() in raw.lower() or raw.lower() in theme.lower():
            return theme
    return None


def coherence(topics_df: pd.DataFrame) -> None:
    """NPMI and UMass coherence: BERTopic topics vs LDA K=174 baseline."""
    log.info("coherence — loading fit-sample docs")
    fit_keys = set(pd.read_parquet(PROC / "conf_universe.parquet").query("in_fit")["dblp_key"])
    pf = pq.ParquetFile(INTER / "conf_enriched.parquet")
    rows = pd.concat(
        [pf.read_row_group(i, ["dblp_key", "text"]).to_pandas() for i in range(pf.num_row_groups)]
    ).drop_duplicates("dblp_key")
    docs = rows[rows["dblp_key"].isin(fit_keys)].dropna(subset=["text"])["text"].tolist()
    D    = len(docs)

    words = [list(r["top_words"])[:10] for _, r in topics_df.iterrows()]
    vocab = sorted({w for ws in words for w in ws})
    w2i   = {w: i for i, w in enumerate(vocab)}
    df_b, co_b = _cooccur(docs, vocab)
    npmis   = [_npmi(w, D, df_b, co_b, w2i) for w in words]
    umasses = [_umass(w, df_b, co_b, w2i)   for w in words]

    comp_f, voc_f = OUT / "lda_components.npy", OUT / "lda_vocab.npy"
    if comp_f.exists():
        comp, lda_vocab = np.load(comp_f), np.load(voc_f, allow_pickle=True)
    else:
        vec = CountVectorizer(
            tokenizer=lemmatizing_tokenizer, token_pattern=None,
            stop_words=load_stopwords(), max_features=20_000, min_df=5,
        )
        Xlda      = vec.fit_transform(docs)
        lda_vocab = np.array(vec.get_feature_names_out())
        comp      = LatentDirichletAllocation(
            n_components=174, random_state=SEED, max_iter=30, n_jobs=-1, learning_method="batch"
        ).fit(Xlda).components_
        np.save(comp_f, comp)
        np.save(voc_f, lda_vocab)

    lda_ws    = [[lda_vocab[i] for i in row.argsort()[:-11:-1]] for row in comp]
    lda_v     = sorted({w for ws in lda_ws for w in ws})
    lda_w2i   = {w: i for i, w in enumerate(lda_v)}
    df_l, co_l = _cooccur(docs, lda_v)
    lda_npmis   = [_npmi(ws, D, df_l, co_l, lda_w2i) for ws in lda_ws]
    lda_umasses = [_umass(ws, df_l, co_l, lda_w2i)   for ws in lda_ws]

    stats = {
        "bertopic": {"npmi_mean": float(np.nanmean(npmis)),     "npmi_std": float(np.nanstd(npmis)),
                     "umass_mean": float(np.nanmean(umasses))},
        "lda_k174": {"npmi_mean": float(np.nanmean(lda_npmis)), "npmi_std": float(np.nanstd(lda_npmis)),
                     "umass_mean": float(np.nanmean(lda_umasses))},
        "n_docs": D, "n_topics": len(topics_df),
    }
    (OUT / "coherence.json").write_text(json.dumps(stats, indent=2))
    log.info("  BERTopic NPMI=%.3f±%.3f  LDA NPMI=%.3f±%.3f",
             stats["bertopic"]["npmi_mean"], stats["bertopic"]["npmi_std"],
             stats["lda_k174"]["npmi_mean"], stats["lda_k174"]["npmi_std"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (ours, ldas, label) in zip(axes, [(npmis, lda_npmis, "NPMI"), (umasses, lda_umasses, "UMass")]):
        data = [[v for v in vals if not np.isnan(v)] for vals in [ours, ldas]]
        vp = ax.violinplot(data, positions=[1, 2], showmedians=True)
        for pc, col in zip(vp["bodies"], ["#2563eb", "#ea580c"]):
            pc.set_facecolor(col)
            pc.set_alpha(0.65)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["BERTopic", "LDA K=174"])
        ax.set_title(label)
        ax.set_ylabel(label)
    fig.suptitle(f"Topic coherence — fit corpus (n={D:,}, {len(topics_df)} topics)")
    _savefig(fig, "coherence")


def accuracy() -> None:
    """Theme assignment accuracy and human/LLM reviewer agreement on the ICSE sample."""
    log.info("accuracy")
    llm   = pd.read_csv(PROC / "claude_review_results_conf_icse.csv")
    human = pd.read_csv(PROC / "human_review_results_conf_icse.csv")

    llm["correct_theme"] = llm["correct"].str.strip().str.lower() == "y"
    llm["suggested"] = llm.apply(
        lambda r: _extract_theme(r["notes"]) if not r["correct_theme"] else None, axis=1
    )
    et = human["error_type"].fillna("").str.strip().str.lower()
    human["correct_theme"] = (human["correct"].str.strip().str.lower() == "y") | (et == "t")
    human["correct_topic"]  = (human["correct"].str.strip().str.lower() == "y") & (et == "")

    merged = llm.merge(human, on="dblp_key", suffixes=("_l", "_h"))
    y_l, y_h = merged["correct_theme_l"].tolist(), merged["correct_theme_h"].tolist()
    pct   = float(np.mean(np.array(y_l) == np.array(y_h)))
    p_e   = np.mean(y_l) * np.mean(y_h) + (1 - np.mean(y_l)) * (1 - np.mean(y_h))
    kappa = float((pct - p_e) / (1 - p_e))
    ac1   = _gwet(y_l, y_h)

    n_llm, k_llm = len(llm),   int(llm["correct_theme"].sum())
    n_h,   k_h   = len(human), int(human["correct_theme"].sum())

    rows = []
    for theme, g in llm.groupby("group"):
        k, n = int(g["correct_theme"].sum()), len(g)
        lo, hi = _wilson(k, n)
        rows.append({"theme": theme, "n": n, "acc": k / n, "ci_lo": lo, "ci_hi": hi})
    theme_acc = pd.DataFrame(rows).sort_values("acc", ascending=False)

    stats = {
        "llm":   {"n": n_llm, "n_wrong": n_llm - k_llm, "accuracy": k_llm / n_llm,
                  "ci": list(_wilson(k_llm, n_llm))},
        "human": {"n": n_h, "n_wrong": n_h - k_h, "accuracy": k_h / n_h,
                  "ci": list(_wilson(k_h, n_h)), "topic_accuracy": float(human["correct_topic"].mean())},
        "overlap_n": len(merged), "pct_agree": pct, "cohen_kappa": kappa, "gwet_ac1": ac1,
        "n_needed_per_theme_5pct": int(np.ceil(1.96**2 * 0.74 * 0.26 / 0.05**2)),
    }
    (OUT / "agreement.json").write_text(json.dumps(stats, indent=2))
    log.info("  LLM %.1f%%  Human %.1f%%  AC1=%.3f  κ=%.3f",
             k_llm / n_llm * 100, k_h / n_h * 100, ac1, kappa)

    pal = _palette()
    ta  = theme_acc.copy()
    ta["_ord"] = ta["theme"].apply(lambda t: list(pal).index(t) if t in pal else 99)
    ta  = ta.sort_values("_ord")
    xs  = np.arange(len(ta))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(xs, ta["acc"], color=[pal.get(t, "#888") for t in ta["theme"]], alpha=0.85, width=0.6)
    ax.errorbar(xs, ta["acc"],
                yerr=[ta["acc"] - ta["ci_lo"], ta["ci_hi"] - ta["acc"]],
                fmt="none", color="black", capsize=4, linewidth=1.2)
    for x, (_, r) in zip(xs, ta.iterrows()):
        ax.text(x, r["ci_hi"] + 0.03, f"n={r['n']}", ha="center", va="bottom", fontsize=8)
    ax.axhline(k_h / n_h, color="gray", linestyle="--", linewidth=1.2,
               label=f"Human theme acc {k_h / n_h:.0%}")
    ax.set_xticks(xs)
    ax.set_xticklabels([_ABBREV.get(t, t) for t in ta["theme"]], rotation=28, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Theme accuracy (LLM reviewer)")
    ax.legend(fontsize=9)
    ax.set_title(f"ICSE per-theme accuracy  (LLM n={n_llm}, Wilson 95% CI)\n"
                 f"Overall: LLM {k_llm / n_llm:.0%} · Human {k_h / n_h:.0%} · AC1={ac1:.2f} · κ={kappa:.2f}")
    _savefig(fig, "accuracy")


def coverage() -> None:
    """Nearest-centroid cosine similarity distribution for all fit-sample papers."""
    log.info("coverage")
    emb       = np.load(PROC / "conf_fit_emb.npy")
    centroids = np.load(PROC / "conf_topic_centroids.npy")
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-10)
    sims = (emb @ centroids.T).max(axis=1)

    taus  = [0.20, 0.30, 0.40]
    stats = {
        "n": len(sims), "scope": "all_fit",
        "median": float(np.median(sims)),
        "p10": float(np.percentile(sims, 10)), "p90": float(np.percentile(sims, 90)),
        "iqr": [float(np.percentile(sims, 25)), float(np.percentile(sims, 75))],
        **{f"frac_below_{t}": float((sims < t).mean()) for t in taus},
    }
    (OUT / "coverage.json").write_text(json.dumps(stats, indent=2))
    log.info("  median=%.3f  p10=%.3f  p90=%.3f  below τ=0.40: %.1f%%",
             stats["median"], stats["p10"], stats["p90"], stats["frac_below_0.4"] * 100)

    sorted_s = np.sort(sims)
    cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sorted_s, cdf, color="#2563eb", linewidth=1.8)
    for tau in taus:
        frac = float((sims < tau).mean())
        ax.axvline(tau, color="gray", linestyle="--", linewidth=1, alpha=0.8)
        ax.annotate(f"τ={tau}\n{frac * 100:.1f}% below",
                    xy=(tau, frac), xytext=(tau - 0.17, frac + 0.07),
                    fontsize=8.5, color="gray",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
    ax.set_xlabel("Nearest-centroid cosine similarity")
    ax.set_ylabel("Cumulative fraction of fit-sample papers")
    ax.set_title(f"Assignment confidence ECDF — all conferences  (n={len(sims):,} fit-sample papers)")
    _savefig(fig, "similarity")


def trend_faithfulness() -> None:
    """Theme-share streamgraph, narrative inflections, and Table 3 milestone onsets."""
    log.info("trend faithfulness")
    tg  = pd.read_parquet(PROC / "conf_topic_groups.parquet")
    pal = dict(zip(tg["group"], tg["color"]))
    ORDER = list(tg.sort_values("order")["group"])

    pt = pd.read_parquet(PROC / "conf_paper_topics.parquet", columns=["year", "topic_id", "group"])
    pt["period"] = (pt["year"] // 5 * 5).astype(int)
    period_n = pt.groupby("period").size()

    # Narrative inflections: AI-for-SE and Mobile landmarks quoted in §5.4 text.
    # Restricted to 1990+ and periods with ≥5K papers to suppress sparse-era noise.
    pt90     = pt[pt["year"].between(1990, 2026)]
    pn90     = pt90.groupby("period").size()
    share90  = (
        pt90.groupby(["period", "group"]).size()
        .div(pn90, level="period")
        .reset_index(name="share")
    )
    valid = pn90[pn90 >= 5_000].index
    inflections = {}
    for group, label in [("AI for Software Engineering", "AI / Neural Networks"),
                          ("Emerging Platforms",          "Mobile / Emerging")]:
        sub   = share90[(share90["group"] == group) & (share90["period"].isin(valid))].sort_values("period")
        delta = sub["share"].diff()
        if delta.notna().any():
            best = sub.loc[delta.idxmax()]
            prev = sub.iloc[sub.index.get_loc(delta.idxmax()) - 1]
            inflections[label] = {
                "period": int(best["period"]),
                "before": round(float(prev["share"]), 4),
                "after":  round(float(best["share"]), 4),
                "slope":  round(float(delta.max()), 4),
            }
            log.info("  %s: %d (%.1f%% → %.1f%%)", label,
                     inflections[label]["period"],
                     inflections[label]["before"] * 100, inflections[label]["after"] * 100)
    (OUT / "inflections.json").write_text(json.dumps(inflections, indent=2))

    # Table 3 milestone onset detection.
    # selector is a group name (str) or a list of topic_ids.
    # Onset = period of maximum positive 5-year delta in share.
    MILESTONES = [
        # key             name                           selector                                     min_year  expected
        ("oop",           "Object-Oriented Programming", "Developer Tooling",                         1975,     "mid-late 1980s"),
        ("formal_methods","Formal Methods",              [48, 83],                                    1975,     "late 1980s"),
        ("security",      "Security",                    [0, 54, 49, 139, 56, 98, 135, 79, 143, 166], 1990,     "from 2000"),
        ("data_mining",   "Data Mining",                 [96, 25, 156],                               1990,     "from 2000"),
        ("cloud",         "Cloud Computing",             [133, 14],                                   1990,     "2006-2010"),
    ]
    milestones = {}
    for key, name, selector, min_year, expected in MILESTONES:
        if isinstance(selector, str):
            counts = pt[pt["group"] == selector].groupby("period").size()
        else:
            counts = pt[pt["topic_id"].isin(selector)].groupby("period").size()
        counts   = counts.reindex(period_n.index, fill_value=0)
        share    = counts / period_n
        sub      = share[share.index >= min_year]
        detected = int(sub.diff().idxmax())
        before   = float(sub.get(detected - 5, float("nan")))
        after    = float(sub.get(detected,     float("nan")))
        milestones[key] = {"name": name, "detected": detected, "expected": expected,
                           "before": round(before, 4), "after": round(after, 4),
                           "slope":  round(after - before, 4)}
        log.info("  %-30s %d  (%.1f%% → %.1f%%)  expected=%s",
                 name, detected, before * 100, after * 100, expected)
    (OUT / "milestones.json").write_text(json.dumps(milestones, indent=2))

    # Streamgraph figure (Figure X in §5.4)
    pivot = share90.pivot(index="period", columns="group", values="share").fillna(0)
    cols  = [c for c in ORDER if c in pivot.columns] + [c for c in pivot.columns if c not in ORDER]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    base = np.zeros(len(pivot))
    for col in pivot.columns:
        v = pivot[col].values.astype(float)
        ax.fill_between(pivot.index, base, base + v, alpha=0.85, color=pal.get(col, "#888"))
        base += v
    ax.set_xlim(1990, 2026)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Period start (year)")
    ax.set_ylabel("Share of all papers")
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.arange(0, 1.1, 0.2)])
    ax.set_title(f"Theme share 1990–2026 — all conferences  (n={len(pt90):,} papers, 5-year periods)")
    handles = [mpatches.Patch(color=pal.get(c, "#888"), label=_ABBREV.get(c, c)) for c in pivot.columns]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left")
    _savefig(fig, "trends")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    topics_df = pd.read_parquet(PROC / "conf_topics.parquet")
    topics_df = topics_df[topics_df["topic_id"] >= 0]

    coherence(topics_df)
    accuracy()
    coverage()
    trend_faithfulness()
    log.info("done — data/validation/  outputs/figures/val_*.png")


if __name__ == "__main__":
    main()
