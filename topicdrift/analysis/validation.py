"""
Validation tasks A–D for the TopicDrift paper.

Run:   make validate   (or python -m topicdrift.analysis.validation)
Data:  data/validation/
Figs:  outputs/figures/val_*.{svg,png}
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
OUT = ROOT / "data/validation"
FIGS = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)
ICSE = "conf/icse"
SEED = 42

# ── palette + figure helpers ──────────────────────────────────────────────────


def _palette() -> dict[str, str]:
    reg = pd.read_parquet(PROC / "conf_topic_groups.parquet")
    return dict(zip(reg["group"], reg["color"]))


_ABBREV = {
    "AI for Software Engineering": "AI for SE",
    "Human Factors in Software Engineering": "Human Factors",
    "Requirements Engineering": "Req. Eng.",
    "Defect Management": "Defect Mgmt",
    "Developer Tooling": "Dev Tooling",
    "Program Correctness": "Prog. Correct.",
}


def _ab(theme: str) -> str:
    return _ABBREV.get(theme, theme)


def _save(fig, name: str) -> None:
    for ext, kw in [(".svg", {}), (".png", {"dpi": 300})]:
        fig.savefig(FIGS / f"val_{name}{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "legend.frameon": False,
        }
    )


# ── A: topic coherence ────────────────────────────────────────────────────────


def _fit_docs() -> list[str]:
    """Text for all 120K in-fit papers (used by Task A coherence + LDA)."""
    keys = set(pd.read_parquet(PROC / "conf_universe.parquet").query("in_fit")["dblp_key"])
    pf = pq.ParquetFile(INTER / "conf_enriched.parquet")
    rows = pd.concat(
        [pf.read_row_group(i, ["dblp_key", "text"]).to_pandas() for i in range(pf.num_row_groups)],
    ).drop_duplicates("dblp_key")
    return rows[rows["dblp_key"].isin(keys)].dropna(subset=["text"])["text"].tolist()


def _cooccur(docs: list[str], vocab: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """doc-freqs and full co-occurrence matrix via a single X.T @ X."""
    X = CountVectorizer(
        tokenizer=lemmatizing_tokenizer,
        token_pattern=None,
        vocabulary={w: i for i, w in enumerate(vocab)},
        binary=True,
    ).fit_transform(docs)
    return (
        np.asarray(X.sum(0), dtype=np.float64).flatten(),
        (X.T @ X).toarray().astype(np.float64),
    )


def _npmi(words: list[str], D: int, df: np.ndarray, co: np.ndarray, w2i: dict) -> float:
    idx = [w2i[w] for w in words if w in w2i]
    if len(idx) < 2:
        return float("nan")
    s = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            dij = co[idx[i], idx[j]]
            if dij == 0:
                s.append(-1.0)
                continue
            pi, pj, pij = df[idx[i]] / D, df[idx[j]] / D, dij / D
            pmi = np.log2(pij) - np.log2(pi) - np.log2(pj)
            s.append(pmi / -np.log2(pij))
    return float(np.mean(s))


def _umass(words: list[str], df: np.ndarray, co: np.ndarray, w2i: dict) -> float:
    idx = [w2i[w] for w in words if w in w2i]
    if len(idx) < 2:
        return float("nan")
    return float(
        np.mean(
            [
                np.log((co[idx[i], idx[j]] + 1) / max(df[idx[j]], 1))
                for i in range(1, len(idx))
                for j in range(i)
            ]
        )
    )


def task_a(topics_df: pd.DataFrame) -> dict:
    """Coherence (NPMI, UMass) for 174 BERTopic topics vs LDA baseline."""
    log.info("A: coherence — loading all %d fit-sample docs...", 120_051)
    docs = _fit_docs()
    D = len(docs)

    vocab = sorted({w for ws in topics_df["top_words"] for w in list(ws)[:10]})
    w2i = {w: i for i, w in enumerate(vocab)}
    df_b, co_b = _cooccur(docs, vocab)
    npmis = [_npmi(list(r["top_words"])[:10], D, df_b, co_b, w2i) for _, r in topics_df.iterrows()]
    umasses = [_umass(list(r["top_words"])[:10], df_b, co_b, w2i) for _, r in topics_df.iterrows()]

    # LDA baseline — cache components to avoid refitting
    comp_f, voc_f = OUT / "lda_components.npy", OUT / "lda_vocab.npy"
    if comp_f.exists():
        comp, lda_vocab = np.load(comp_f), np.load(voc_f, allow_pickle=True)
    else:
        vec = CountVectorizer(
            tokenizer=lemmatizing_tokenizer,
            token_pattern=None,
            stop_words=load_stopwords(),
            max_features=20_000,
            min_df=5,
        )
        Xlda = vec.fit_transform(docs)
        lda_vocab = np.array(vec.get_feature_names_out())
        comp = (
            LatentDirichletAllocation(
                n_components=174, random_state=SEED, max_iter=30, n_jobs=-1, learning_method="batch"
            )
            .fit(Xlda)
            .components_
        )
        np.save(comp_f, comp)
        np.save(voc_f, lda_vocab)

    lda_ws = [[lda_vocab[i] for i in row.argsort()[:-11:-1]] for row in comp]
    lda_v = sorted({w for ws in lda_ws for w in ws})
    lda_w2i = {w: i for i, w in enumerate(lda_v)}
    df_l, co_l = _cooccur(docs, lda_v)
    lda_npmis = [_npmi(ws, D, df_l, co_l, lda_w2i) for ws in lda_ws]
    lda_umasses = [_umass(ws, df_l, co_l, lda_w2i) for ws in lda_ws]

    pd.DataFrame(
        {
            "topic_id": topics_df["topic_id"],
            "label": topics_df["llm_label"],
            "group": topics_df["group"],
            "n_papers": topics_df["size"],
            "npmi": npmis,
            "umass": umasses,
        }
    ).to_csv(OUT / "coherence.csv", index=False)

    stats = {
        "bertopic": {
            "npmi_mean": float(np.nanmean(npmis)),
            "npmi_std": float(np.nanstd(npmis)),
            "umass_mean": float(np.nanmean(umasses)),
        },
        "lda_k174": {
            "npmi_mean": float(np.nanmean(lda_npmis)),
            "npmi_std": float(np.nanstd(lda_npmis)),
            "umass_mean": float(np.nanmean(lda_umasses)),
        },
        "n_docs": D,
        "n_topics": len(topics_df),
    }
    (OUT / "coherence.json").write_text(json.dumps(stats, indent=2))
    log.info(
        "  BERTopic NPMI=%.3f ± %.3f  LDA NPMI=%.3f ± %.3f",
        stats["bertopic"]["npmi_mean"],
        stats["bertopic"]["npmi_std"],
        stats["lda_k174"]["npmi_mean"],
        stats["lda_k174"]["npmi_std"],
    )

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (ours, ldas, metric) in zip(
        axes,
        [
            (npmis, lda_npmis, "NPMI"),
            (umasses, lda_umasses, "UMass"),
        ],
    ):
        data = [[v for v in vals if not np.isnan(v)] for vals in [ours, ldas]]
        vp = ax.violinplot(data, positions=[1, 2], showmedians=True)
        for pc, col in zip(vp["bodies"], ["#2563eb", "#ea580c"]):
            pc.set_facecolor(col)
            pc.set_alpha(0.65)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["BERTopic", "LDA K=174"])
        ax.set_title(metric)
        ax.set_ylabel(metric)
    fig.suptitle(
        f"Topic coherence — all-conference fit corpus (n={D:,} docs, {len(topics_df)} topics)"
    )
    _save(fig, "coherence")
    return stats


# ── B: manual accuracy ────────────────────────────────────────────────────────


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    m = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / (1 + z**2 / n)
    return max(0.0, c - m), min(1.0, c + m)


def _gwet(y1, y2) -> float:
    """Gwet's AC1 — robust to prevalence-induced kappa paradox."""
    a, b = np.array(y1, bool), np.array(y2, bool)
    p_o = (a == b).mean()
    pi = (a.mean() + b.mean()) / 2
    p_e = pi * (1 - pi) + (1 - pi) * pi  # = 2·π·(1-π)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def _extract_theme(notes: str) -> str | None:
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


def task_b() -> dict:
    """Accuracy and reviewer agreement on ICSE annotation sample."""
    log.info("B: accuracy")
    llm = pd.read_csv(ROOT / "data/processed/claude_review_results_conf_icse.csv")
    human = pd.read_csv(ROOT / "data/processed/human_review_results_conf_icse.csv")

    llm["correct_theme"] = llm["correct"].str.strip().str.lower() == "y"
    llm["suggested"] = llm.apply(
        lambda r: _extract_theme(r["notes"]) if not r["correct_theme"] else None, axis=1
    )

    et = human["error_type"].fillna("").str.strip().str.lower()
    human["correct_theme"] = (human["correct"].str.strip().str.lower() == "y") | (et == "t")
    human["correct_topic"] = (human["correct"].str.strip().str.lower() == "y") & (et == "")

    merged = llm.merge(human, on="dblp_key", suffixes=("_l", "_h"))
    y_l = merged["correct_theme_l"].tolist()
    y_h = merged["correct_theme_h"].tolist()
    pct = float(np.mean(np.array(y_l) == np.array(y_h)))
    kappa = float(
        (pct - (np.mean(y_l) * np.mean(y_h) + (1 - np.mean(y_l)) * (1 - np.mean(y_h))))
        / (1 - (np.mean(y_l) * np.mean(y_h) + (1 - np.mean(y_l)) * (1 - np.mean(y_h))))
    )
    ac1 = _gwet(y_l, y_h)

    n_llm, k_llm = len(llm), int(llm["correct_theme"].sum())
    n_h, k_h = len(human), int(human["correct_theme"].sum())

    # Per-theme accuracy (LLM)
    rows = []
    for theme, g in llm.groupby("group"):
        k, n = int(g["correct_theme"].sum()), len(g)
        lo, hi = _wilson(k, n)
        rows.append({"theme": theme, "n": n, "acc": k / n, "ci_lo": lo, "ci_hi": hi})
    theme_acc = pd.DataFrame(rows).sort_values("acc", ascending=False)
    theme_acc.to_csv(OUT / "accuracy.csv", index=False)

    # Confusion: assigned → suggested (for the 26 LLM-wrong papers)
    wrong = llm[~llm["correct_theme"]].copy()
    conf_counts = wrong.groupby(["group", "suggested"]).size().reset_index(name="n")
    conf_counts.to_csv(OUT / "confusion.csv", index=False)

    stats = {
        "llm": {
            "n": n_llm,
            "n_wrong": n_llm - k_llm,
            "accuracy": k_llm / n_llm,
            "ci": list(_wilson(k_llm, n_llm)),
        },
        "human": {
            "n": n_h,
            "n_wrong": n_h - k_h,
            "accuracy": k_h / n_h,
            "ci": list(_wilson(k_h, n_h)),
            "topic_accuracy": float(human["correct_topic"].mean()),
        },
        "overlap_n": len(merged),
        "pct_agree": pct,
        "cohen_kappa": kappa,
        "gwet_ac1": ac1,
        "kappa_note": (
            f"κ≈0 is the prevalence paradox (avg 'correct' rate={np.mean([np.mean(y_l), np.mean(y_h)]):.2f}). "
            f"AC1={ac1:.3f} is the appropriate statistic."
        ),
        "n_needed_per_theme_5pct": int(np.ceil(1.96**2 * 0.74 * 0.26 / 0.05**2)),
    }
    (OUT / "agreement.json").write_text(json.dumps(stats, indent=2))
    log.info(
        "  LLM acc=%.1f%%  Human theme acc=%.1f%%  AC1=%.3f  κ=%.3f",
        k_llm / n_llm * 100,
        k_h / n_h * 100,
        ac1,
        kappa,
    )

    pal = _palette()
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ta = theme_acc.copy()
    ta["_ord"] = ta["theme"].apply(lambda t: list(pal).index(t) if t in pal else 99)
    ta = ta.sort_values("_ord")
    xs = np.arange(len(ta))
    ax.bar(xs, ta["acc"], color=[pal.get(t, "#888") for t in ta["theme"]], alpha=0.85, width=0.6)
    ax.errorbar(
        xs,
        ta["acc"],
        yerr=[ta["acc"] - ta["ci_lo"], ta["ci_hi"] - ta["acc"]],
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.2,
    )
    for x, (_, r) in zip(xs, ta.iterrows()):
        ax.text(x, r["ci_hi"] + 0.03, f"n={r['n']}", ha="center", va="bottom", fontsize=8)
    ax.axhline(
        k_h / n_h,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Human theme acc {k_h / n_h:.0%}",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([_ab(t) for t in ta["theme"]], rotation=28, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Theme accuracy (LLM reviewer)")
    ax.legend(fontsize=9)
    ax.set_title(
        f"ICSE per-theme accuracy  (LLM n={n_llm}, Wilson 95% CI)\n"
        f"Overall: LLM {k_llm / n_llm:.0%} · Human {k_h / n_h:.0%} · AC1={ac1:.2f} · κ={kappa:.2f}"
    )
    _save(fig, "accuracy")
    return stats


# ── C: assignment coverage ────────────────────────────────────────────────────


def task_c() -> dict:
    """Nearest-centroid cosine similarity for all 120K fit-sample papers."""
    log.info("C: coverage — computing similarities for full fit sample")
    emb = np.load(PROC / "conf_fit_emb.npy")  # (120K, 384) unit-norm
    centroids = np.load(PROC / "conf_topic_centroids.npy")
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-10)
    sims = (emb @ centroids.T).max(axis=1)

    taus = [0.20, 0.30, 0.40]
    stats = {
        "n": len(sims),
        "scope": "all_fit",
        "median": float(np.median(sims)),
        "p10": float(np.percentile(sims, 10)),
        "p90": float(np.percentile(sims, 90)),
        "iqr": [float(np.percentile(sims, 25)), float(np.percentile(sims, 75))],
        **{f"frac_below_{t}": float((sims < t).mean()) for t in taus},
    }
    (OUT / "coverage.json").write_text(json.dumps(stats, indent=2))
    log.info(
        "  median=%.3f  p10=%.3f  p90=%.3f  below τ=0.40: %.1f%%",
        stats["median"],
        stats["p10"],
        stats["p90"],
        stats["frac_below_0.4"] * 100,
    )

    # ECDF figure with tau thresholds annotated
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sorted_s = np.sort(sims)
    cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax.plot(sorted_s, cdf, color="#2563eb", linewidth=1.8)
    for tau in taus:
        frac = float((sims < tau).mean())
        ax.axvline(tau, color="gray", linestyle="--", linewidth=1, alpha=0.8)
        ax.annotate(
            f"τ={tau}\n{frac * 100:.1f}% below",
            xy=(tau, frac),
            xytext=(tau - 0.17, frac + 0.07),
            fontsize=8.5,
            color="gray",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )
    ax.set_xlabel("Nearest-centroid cosine similarity")
    ax.set_ylabel("Cumulative fraction of fit-sample papers")
    ax.set_title(
        f"Assignment confidence ECDF — all conferences  (n={len(sims):,} fit-sample papers)"
    )
    _save(fig, "similarity")
    return stats


# ── D: trend faithfulness ─────────────────────────────────────────────────────


def task_d() -> dict:
    """Theme shares by 5-year period across all conferences + streamgraph figure."""
    log.info("D: trends — all conferences")
    pt = pd.read_parquet(PROC / "conf_paper_topics.parquet", columns=["year", "group"])
    pt = pt[pt["year"].between(1990, 2026)].copy()
    pt["period"] = (pt["year"] // 5 * 5).astype(int)

    period_n = pt.groupby("period").size()
    share = (
        pt.groupby(["period", "group"])
        .size()
        .div(period_n, level="period")
        .reset_index(name="share")
    )
    share.to_csv(OUT / "trends.csv", index=False)

    # Inflection detection (min 5K papers per period to avoid sparse noise)
    MIN_N = 5_000
    valid_periods = period_n[period_n >= MIN_N].index
    inflections = {}
    LANDMARKS = {
        "AI for Software Engineering": "AI / Neural Networks",
        "Emerging Platforms": "Mobile / Emerging",
    }
    for group, label in LANDMARKS.items():
        sub = share[(share["group"] == group) & (share["period"].isin(valid_periods))].sort_values(
            "period"
        )
        delta = sub["share"].diff()
        if delta.notna().any():
            best = sub.loc[delta.idxmax()]
            prev = sub.iloc[sub.index.get_loc(delta.idxmax()) - 1]
            inflections[label] = {
                "period": int(best["period"]),
                "before": round(float(prev["share"]), 4),
                "after": round(float(best["share"]), 4),
                "slope": round(float(delta.max()), 4),
            }
    (OUT / "inflections.json").write_text(json.dumps(inflections, indent=2))
    for label, inf in inflections.items():
        log.info(
            "  %s: inflection at %d (%.1f%% → %.1f%%)",
            label,
            inf["period"],
            inf["before"] * 100,
            inf["after"] * 100,
        )

    # Streamgraph figure
    pal = _palette()
    ORDER = list(pd.read_parquet(PROC / "conf_topic_groups.parquet").sort_values("order")["group"])
    pivot = share.pivot(index="period", columns="group", values="share").fillna(0)
    cols = [c for c in ORDER if c in pivot.columns] + [c for c in pivot.columns if c not in ORDER]
    pivot = pivot[cols]

    _style()
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    base = np.zeros(len(pivot))
    for col in pivot.columns:
        v = pivot[col].values.astype(float)
        ax.fill_between(
            pivot.index, base, base + v, alpha=0.85, color=pal.get(col, "#888"), label=_ab(col)
        )
        base += v
    ax.set_xlim(1990, 2026)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Period start (year)")
    ax.set_ylabel("Share of all papers")
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.arange(0, 1.1, 0.2)])
    ax.set_title(f"Theme share 1990–2026 — all conferences  (n={len(pt):,} papers, 5-year periods)")
    handles = [mpatches.Patch(color=pal.get(c, "#888"), label=_ab(c)) for c in pivot.columns]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left")
    _save(fig, "trends")
    return {"n_papers": len(pt), "share_rows": len(share), "inflections": inflections}


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    topics_df = pd.read_parquet(PROC / "conf_topics.parquet")
    topics_df = topics_df[topics_df["topic_id"] >= 0]

    task_a(topics_df)
    task_b()
    task_c()
    task_d()
    log.info("done — data/validation/  outputs/figures/val_*")


if __name__ == "__main__":
    main()
