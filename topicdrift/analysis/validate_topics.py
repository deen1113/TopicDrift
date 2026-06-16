"""
validate_topics.py — Quantify topic & theme coherence for the global model.

Reporting companion to the pipeline. Produces the numbers the paper's
Validation section needs, computed from the saved fit embeddings + centroids
(no re-fit, no re-embed required):

  Topic-level   silhouette (cosine), per-topic cohesion and margin to the
                nearest other topic, plus the weakest topics by name.
  Theme-level   silhouette of the hand-curated 10 themes over the 174 topic
                centroids, per-theme spread, and any "misfiled" topics whose
                centroid is nearest a theme other than the one assigned.
  Word-level    topic diversity (unique top-words) and redundancy (mean
                pairwise Jaccard of top words).
  External      milestone onset hit-rate read from topic_sanity_events.csv.

Embedding metrics are evaluated on the fit sample (the population the
clustering was actually fit on); assignment to that population is exact
(nearest-centroid agreement is 100%), so centroid-based scores are faithful.

Reads (default --processed data2/processed):
  conf_universe.parquet, conf_paper_topics.parquet, conf_topics.parquet,
  conf_topic_centroids.npy, conf_fit_emb.npy
  outputs/tables/topic_sanity_events.csv  (optional)
Writes:
  outputs/tables/topic_validation_summary.csv
  outputs/tables/topic_validation_topics.csv
  outputs/tables/topic_validation_themes.csv
  outputs/tables/topic_validation.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

log = logging.getLogger(__name__)

OUTPUTS_TABLES = Path("outputs/tables")
TOP_N_WORDS = 10
SILHOUETTE_SAMPLE = 12000  # subsample for the O(n^2) topic silhouette
WEAKEST_K = 10             # how many worst topics to list
SEED = 42


# ── loading ───────────────────────────────────────────────────────────────


def _unit(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows so dot products are cosine similarities."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def load_model(processed: Path):
    topics = pd.read_parquet(processed / "conf_topics.parquet")
    uni = pd.read_parquet(processed / "conf_universe.parquet")
    pt = pd.read_parquet(processed / "conf_paper_topics.parquet")[["dblp_key", "topic_id"]]
    centroids = np.load(processed / "conf_topic_centroids.npy")
    emb = np.load(processed / "conf_fit_emb.npy")

    fit = uni[uni["in_fit"]].reset_index(drop=True).merge(pt, on="dblp_key", how="left")
    if len(fit) != len(emb):
        raise ValueError(f"fit rows ({len(fit)}) != embeddings ({len(emb)}) — order mismatch")
    if fit["topic_id"].isna().any():
        raise ValueError("some fit papers have no topic assignment")

    labels = fit["topic_id"].to_numpy(dtype=int)
    return topics, centroids, _unit(emb), labels


# ── topic-level coherence ───────────────────────────────────────────────────


def topic_metrics(topics: pd.DataFrame, centroids: np.ndarray,
                  emb_unit: np.ndarray, labels: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Per-topic cohesion (mean cosine of members to own centroid) and margin
    to the nearest other topic centroid; plus the overall silhouette."""
    cen = _unit(centroids)
    n_topics = cen.shape[0]

    # cohesion: mean cosine(member, own centroid), vectorised over all members
    own_sim = np.einsum("ij,ij->i", emb_unit, cen[labels])
    cohesion = pd.Series(own_sim).groupby(labels).mean()

    # nearest other topic by centroid-centroid cosine
    cc = cen @ cen.T
    np.fill_diagonal(cc, -np.inf)
    nearest = cc.argmax(axis=1)
    nearest_sim = cc[np.arange(n_topics), nearest]

    rows = []
    label_by_id = dict(zip(topics["topic_id"], topics["llm_label"]))
    group_by_id = dict(zip(topics["topic_id"], topics["group"]))
    size_by_id = dict(zip(topics["topic_id"], topics["size"]))
    for t in range(n_topics):
        coh = float(cohesion.get(t, float("nan")))
        rows.append({
            "topic_id": t,
            "label": label_by_id.get(t, f"topic {t}"),
            "group": group_by_id.get(t, ""),
            "size": int(size_by_id.get(t, 0)),
            "cohesion": round(coh, 3),
            "nearest_topic": label_by_id.get(nearest[t], f"topic {nearest[t]}"),
            "nearest_topic_sim": round(float(nearest_sim[t]), 3),
            "margin": round(coh - float(nearest_sim[t]), 3),
        })
    per_topic = pd.DataFrame(rows).sort_values("cohesion")

    # overall silhouette on a subsample (cosine)
    rng = np.random.default_rng(SEED)
    n = len(labels)
    idx = rng.choice(n, min(SILHOUETTE_SAMPLE, n), replace=False)
    sil = float(silhouette_score(emb_unit[idx], labels[idx], metric="cosine", random_state=SEED))

    summary = {
        "n_topics": n_topics,
        "topic_silhouette_cosine": round(sil, 3),
        "mean_topic_cohesion": round(float(per_topic["cohesion"].mean()), 3),
        "mean_topic_margin": round(float(per_topic["margin"].mean()), 3),
        "n_topics_margin_below_0": int((per_topic["margin"] < 0).sum()),
    }
    return per_topic, summary


# ── theme-level coherence ───────────────────────────────────────────────────


def theme_metrics(topics: pd.DataFrame, centroids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Treat each of the 174 topic centroids as a point labelled by its theme.
    Silhouette over those points scores how well the hand-curated themes
    separate. A topic is 'misfiled' if its centroid is nearest the prototype
    (mean centroid) of a theme other than the one it was grouped into."""
    cen = _unit(centroids)
    themes = topics["group"].astype(str).to_numpy()
    uniq = sorted(pd.unique(themes))
    theme_id = {g: i for i, g in enumerate(uniq)}
    y = np.array([theme_id[g] for g in themes])

    theme_sil = float(silhouette_score(cen, y, metric="cosine")) if len(uniq) > 1 else float("nan")

    # theme prototype = mean of its topics' centroids (re-normalised)
    protos = _unit(np.vstack([cen[y == i].mean(axis=0) for i in range(len(uniq))]))
    topic_to_proto = cen @ protos.T          # (n_topics, n_themes)
    nearest_theme = topic_to_proto.argmax(axis=1)
    misfiled_mask = nearest_theme != y

    rows = []
    for i, g in enumerate(uniq):
        members = np.where(y == i)[0]
        intra = cen[members] @ protos[i]
        others = [j for j in range(len(uniq)) if j != i]
        nearest_other = protos[i] @ protos[others].T
        no_idx = int(np.argmax(nearest_other))
        mis = [topics.iloc[m]["llm_label"] for m in members if misfiled_mask[m]]
        rows.append({
            "theme": g,
            "n_topics": int(len(members)),
            "intra_theme_sim": round(float(intra.mean()), 3),
            "nearest_theme": uniq[others[no_idx]],
            "nearest_theme_sim": round(float(nearest_other[no_idx]), 3),
            "n_misfiled": int(len(mis)),
            "misfiled_topics": "; ".join(mis[:8]),
        })
    per_theme = pd.DataFrame(rows).sort_values("intra_theme_sim")

    summary = {
        "n_themes": len(uniq),
        "theme_silhouette_cosine": round(theme_sil, 3),
        "mean_theme_intra_sim": round(float(per_theme["intra_theme_sim"].mean()), 3),
        "n_misfiled_topics": int(misfiled_mask.sum()),
        "pct_misfiled_topics": round(100 * float(misfiled_mask.mean()), 1),
    }
    return per_theme, summary


# ── word-level structure ────────────────────────────────────────────────────


def _top_words(topics: pd.DataFrame, n: int) -> list[list[str]]:
    out = []
    for w in topics["top_words"]:
        words = list(w)[:n] if w is not None else []
        out.append([str(x).lower() for x in words])
    return out


def word_metrics(topics: pd.DataFrame) -> dict:
    words = _top_words(topics, TOP_N_WORDS)
    flat = [w for ws in words for w in ws]
    diversity = len(set(flat)) / max(len(flat), 1)

    sets = [set(ws) for ws in words]
    jac, redundant_pairs = [], 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            if not u:
                continue
            s = len(sets[i] & sets[j]) / len(u)
            jac.append(s)
            if s >= 0.2:
                redundant_pairs += 1
    return {
        "topic_diversity_top10": round(diversity, 3),
        "mean_pairwise_jaccard": round(float(np.mean(jac)) if jac else 0.0, 4),
        "n_topic_pairs_jaccard_ge_0.2": int(redundant_pairs),
    }


# ── external milestone validation ───────────────────────────────────────────


def milestone_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    total = len(df)
    ok = int((df["status"].astype(str).str.strip() == "ok").sum())
    no_match = int(df["matched_topic_id"].isna().sum()) if "matched_topic_id" in df else 0
    deltas = pd.to_numeric(df.get("delta_years"), errors="coerce").abs().dropna()
    return {
        "milestones_total": total,
        "milestones_on_window": ok,
        "milestone_hit_rate": round(ok / total, 2) if total else float("nan"),
        "milestones_no_match": no_match,
        "mean_abs_onset_delta_years": round(float(deltas.mean()), 1) if len(deltas) else float("nan"),
    }


# ── report ──────────────────────────────────────────────────────────────────


def write_report(summary: dict, per_topic: pd.DataFrame, per_theme: pd.DataFrame, out: Path):
    lines = ["# Topic & theme coherence — validation report", ""]
    lines.append("Computed by `validate_topics.py` from the saved fit embeddings + "
                 "centroids. Embedding metrics use cosine similarity on the fit sample.")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in summary.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Weakest topics (lowest cohesion)")
    lines.append("")
    lines.append("| Topic | Theme | Size | Cohesion | Nearest topic | Margin |")
    lines.append("|---|---|---:|---:|---|---:|")
    for _, r in per_topic.head(WEAKEST_K).iterrows():
        lines.append(f"| {r.label} | {r.group} | {r['size']} | {r.cohesion} | "
                     f"{r.nearest_topic} | {r.margin} |")
    lines.append("")
    lines.append("## Themes (lowest internal similarity first)")
    lines.append("")
    lines.append("| Theme | #Topics | Intra-sim | Nearest theme | Nearest-sim | #Misfiled |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for _, r in per_theme.iterrows():
        lines.append(f"| {r.theme} | {r.n_topics} | {r.intra_theme_sim} | "
                     f"{r.nearest_theme} | {r.nearest_theme_sim} | {r.n_misfiled} |")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Quantify topic/theme coherence.")
    ap.add_argument("--processed", type=Path, default=Path("data2/processed"),
                    help="dir with conf_* model artifacts (default: data2/processed)")
    ap.add_argument("--sanity", type=Path,
                    default=OUTPUTS_TABLES / "topic_sanity_events.csv")
    args = ap.parse_args()

    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    topics, centroids, emb_unit, labels = load_model(args.processed)

    per_topic, s_topic = topic_metrics(topics, centroids, emb_unit, labels)
    per_theme, s_theme = theme_metrics(topics, centroids)
    s_word = word_metrics(topics)
    s_mile = milestone_metrics(args.sanity)

    summary = {**s_topic, **s_theme, **s_word, **s_mile}

    pd.DataFrame([summary]).to_csv(OUTPUTS_TABLES / "topic_validation_summary.csv", index=False)
    per_topic.to_csv(OUTPUTS_TABLES / "topic_validation_topics.csv", index=False)
    per_theme.to_csv(OUTPUTS_TABLES / "topic_validation_themes.csv", index=False)
    write_report(summary, per_topic, per_theme, OUTPUTS_TABLES / "topic_validation.md")

    log.info("\n=== Coherence summary ===")
    for k, v in summary.items():
        log.info("  %-32s %s", k, v)
    log.info("\nWrote outputs/tables/topic_validation_{summary,topics,themes}.csv + .md")


if __name__ == "__main__":
    main()
