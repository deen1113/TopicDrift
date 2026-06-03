"""
topics_conf.py — ONE global BERTopic fit shared by all website tabs.

Multi-conference site pipeline, step 2/4:
  select_corpus -> topics_conf -> map_seed_themes -> apply_topic_groups

Fit clustering on the stratified sample (select_corpus.py: in_fit=True), take an
L2-normalised embedding centroid per topic, label with the local LLM, then assign
papers to their nearest centroid:
  --assign sample : assign the fit sample only (fast)
  --assign all    : assign the whole 2.3M-paper universe, reusing the saved fit

Reads  conf_universe.parquet, conf_enriched.parquet
Writes conf_topics.parquet, conf_paper_topics.parquet, conf_topic_centroids.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from topicdrift.topic_model import TopicModel, load_stopwords

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
# Silver layer (per-paper rows): build_conf_corpus writes here.
CONF_ENRICHED = INTERIM_DIR / "conf_enriched.parquet"
# Gold layer (analysis outputs):
UNIVERSE = PROCESSED_DIR / "conf_universe.parquet"

TOPICS_OUT = PROCESSED_DIR / "conf_topics.parquet"
PAPER_TOPICS_OUT = PROCESSED_DIR / "conf_paper_topics.parquet"
CENTROIDS_OUT = PROCESSED_DIR / "conf_topic_centroids.npy"
FIT_EMB = PROCESSED_DIR / "conf_fit_emb.npy"

VENUES_CFG = Path("config/venues.yaml")
EMBED_BATCH = 1024
ASSIGN_BATCH = 200_000


def _cfg() -> dict:
    return yaml.safe_load(VENUES_CFG.read_text())


# ── text lookup ───────────────────────────────────────────────────────────────


def load_text(dblp_keys: list[str], cols=("dblp_key", "text", "title")) -> pd.DataFrame:
    """Pull selected columns from conf_enriched for the given keys, in key order."""
    want = set(dblp_keys)
    parts = []
    pf = pq.ParquetFile(CONF_ENRICHED)
    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i, columns=list(cols)).to_pandas()
        hit = rg[rg["dblp_key"].isin(want)]
        if len(hit):
            parts.append(hit)
    df = pd.concat(parts, ignore_index=True).drop_duplicates("dblp_key")
    order = pd.Categorical(df["dblp_key"], categories=dblp_keys, ordered=True)
    return df.assign(_o=order).sort_values("_o").drop(columns="_o").reset_index(drop=True)


# ── embedding ───────────────────────────────────────────────────────────────--


def embed_docs(docs: list[str], embedder) -> np.ndarray:
    """L2-normalised float32 embeddings."""
    emb = embedder.encode(
        docs,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


# ── fit ───────────────────────────────────────────────────────────────────────


def fit_topics(
    fit_df: pd.DataFrame, min_topic_size: int, seed: int, cluster_selection: str = "eom"
):
    """Fit BERTopic on the sample; return (TopicModel, topics_list, embeddings)."""
    stopwords = load_stopwords()
    tm = TopicModel(
        seed=seed,
        min_topic_size=min_topic_size,
        stopwords=stopwords,
        cluster_selection_method=cluster_selection,
    )

    if FIT_EMB.exists() and np.load(FIT_EMB, mmap_mode="r").shape[0] == len(fit_df):
        print(f"  [cache] {FIT_EMB.name}")
        embeddings = np.load(FIT_EMB)
    else:
        print(f"  embedding {len(fit_df):,} fit docs…")
        embeddings = embed_docs(fit_df["text"].tolist(), tm.embedder())
        np.save(FIT_EMB, embeddings)

    print(f"\n=== Global fit (seed={seed}, min_topic_size={min_topic_size}) ===")
    topics = tm.fit(fit_df["text"].tolist(), embeddings=embeddings, reduce_outliers=False)
    n = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"  → {n} topics")
    groups = tm.merge_duplicates(fit_df["text"].tolist())
    if groups:
        topics = tm.topics_
        n = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"  → {n} topics after merging {len(groups)} duplicate groups")
    return tm, list(topics), embeddings


def topic_centroids(embeddings: np.ndarray, topics: list[int]) -> tuple[np.ndarray, list[int]]:
    """Mean (renormalised) embedding per non-outlier topic, ordered by topic_id."""
    topics_arr = np.asarray(topics)
    ids = sorted(t for t in set(topics) if t != -1)
    rows = []
    for tid in ids:
        v = embeddings[topics_arr == tid].mean(axis=0)
        nrm = np.linalg.norm(v)
        rows.append(v / nrm if nrm else v)
    return np.vstack(rows).astype(np.float32), ids


def assign_nearest(emb: np.ndarray, centroids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Argmax cosine (emb and centroids are unit-norm) → topic_id per row."""
    out = np.empty(len(emb), dtype=np.int32)
    for s in range(0, len(emb), ASSIGN_BATCH):
        chunk = np.asarray(emb[s : s + ASSIGN_BATCH], dtype=np.float32)
        out[s : s + len(chunk)] = ids[(chunk @ centroids.T).argmax(axis=1)]
    return out


# ── assignment universes ───────────────────────────────────────────────────────


def assign_sample(fit_df, embeddings, centroids, ids) -> pd.DataFrame:
    tid = assign_nearest(embeddings, centroids, np.asarray(ids))
    return fit_df.assign(topic_id=tid)[["dblp_key", "conf", "year", "topic_id"]]


def assign_all(universe: pd.DataFrame, centroids, ids, embedder) -> pd.DataFrame:
    """Stream conf_enriched once; embed + assign every universe paper by nearest
    centroid. Checkpoints one parquet per row group so a re-run resumes."""
    ids_arr = np.asarray(ids)
    want = set(universe["dblp_key"])
    parts_dir = PROCESSED_DIR / "conf_assign_parts"
    parts_dir.mkdir(exist_ok=True)

    pf = pq.ParquetFile(CONF_ENRICHED)
    n_rg = pf.num_row_groups
    seen = 0
    for i in range(n_rg):
        part = parts_dir / f"part_{i:04d}.parquet"
        if part.exists():
            seen += len(pd.read_parquet(part, columns=["dblp_key"]))
            continue
        rg = pf.read_row_group(i, columns=["dblp_key", "text"]).to_pandas()
        rg = rg[rg["dblp_key"].isin(want)]
        if len(rg):
            emb = embed_docs(rg["text"].tolist(), embedder)
            rg = rg.assign(topic_id=ids_arr[(emb @ centroids.T).argmax(axis=1)])
            seen += len(rg)
        rg[["dblp_key", "topic_id"]].to_parquet(part, index=False)
        print(f"  row group {i + 1}/{n_rg}: {seen:,} assigned so far")

    asg = pd.concat(
        [pd.read_parquet(p) for p in sorted(parts_dir.glob("part_*.parquet"))],
        ignore_index=True,
    ).drop_duplicates("dblp_key")
    out = universe.merge(asg, on="dblp_key", how="inner")
    return out[["dblp_key", "conf", "year", "topic_id"]]


# ── main ───────────────────────────────────────────────────────────────────────


def _write_topic_table(info: pd.DataFrame, sizes: pd.Series) -> pd.DataFrame:
    """conf_topics.parquet: topic_id, llm_label, top_words, size (assigned count)."""
    keep = [c for c in ("topic_id", "llm_label", "top_words") if c in info.columns]
    df = info[info["topic_id"] != -1][keep].copy()
    df["size"] = df["topic_id"].map(sizes).fillna(0).astype(int)
    df = df.sort_values("size", ascending=False).reset_index(drop=True)
    df.to_parquet(TOPICS_OUT, index=False)
    return df


def run_fit(args) -> None:
    """Fit the clustering on the sample, label, and assign (sample or all)."""
    seed = int(_cfg().get("fit", {}).get("seed", 42))
    universe = pd.read_parquet(UNIVERSE)
    fit_keys = universe.loc[universe["in_fit"], "dblp_key"].tolist()
    print(f"Loading text for {len(fit_keys):,} fit docs…")
    fit_df = (
        universe[universe["in_fit"]]
        .merge(load_text(fit_keys), on="dblp_key", how="left")
        .dropna(subset=["text"])
        .reset_index(drop=True)
    )

    tm, topics, embeddings = fit_topics(
        fit_df, args.min_topic_size, seed, cluster_selection=args.cluster_selection
    )
    centroids, ids = topic_centroids(embeddings, topics)
    np.save(CENTROIDS_OUT, centroids)
    print(f"  {len(ids)} topic centroids → {CENTROIDS_OUT.name}")

    info = tm.topic_info()
    if args.no_label:
        info["llm_label"] = info["top_words"].apply(
            lambda w: " ".join(str(x).capitalize() for x in list(w)[:3])
        )
        print(f"  [no-label] {len(ids)} topics (top-word labels)")
    else:
        fit_pt = fit_df.assign(topic_id=topics)[["dblp_key", "topic_id"]]
        info = tm.label_topics_llm(info, fit_pt, fit_df[["dblp_key", "title"]])

    if args.assign == "all":
        paper_topics = assign_all(universe, centroids, ids, tm.embedder())
    else:
        paper_topics = assign_sample(fit_df, embeddings, centroids, ids)
    paper_topics.to_parquet(PAPER_TOPICS_OUT, index=False)
    print(f"  wrote {PAPER_TOPICS_OUT.name} ({len(paper_topics):,} rows, mode={args.assign})")
    df = _write_topic_table(info, paper_topics.groupby("topic_id").size())
    print(f"  wrote {TOPICS_OUT.name} ({len(df)} topics)")


def run_assign_all() -> None:
    """Reuse the saved fit: assign the whole universe and refresh topic sizes.
    Leaves labels and grouping untouched (only counts change)."""
    topics_df = pd.read_parquet(TOPICS_OUT)
    centroids = np.load(CENTROIDS_OUT)
    ids = sorted(topics_df["topic_id"].astype(int))
    if len(ids) != centroids.shape[0]:
        raise SystemExit("centroids out of sync with conf_topics — re-run with --refit")

    universe = pd.read_parquet(UNIVERSE)
    paper_topics = assign_all(universe, centroids, ids, TopicModel().embedder())
    paper_topics.to_parquet(PAPER_TOPICS_OUT, index=False)
    sizes = paper_topics.groupby("topic_id").size()
    topics_df["size"] = topics_df["topic_id"].map(sizes).fillna(0).astype(int)
    topics_df.to_parquet(TOPICS_OUT, index=False)
    print(f"  assigned {len(paper_topics):,} papers to {len(ids)} topics (reused fit)")
    print(
        "  next: `python src/analysis/apply_topic_groups.py --prefix conf_ "
        "--config config/topic_groups.conf.yaml --title 'All Conferences'`"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign", choices=["sample", "all"], default="sample")
    ap.add_argument("--min-topic-size", type=int, default=60)
    ap.add_argument(
        "--cluster-selection",
        choices=["eom", "leaf"],
        default="leaf",
        help="'leaf' gives many fine topics; 'eom' a few broad ones",
    )
    ap.add_argument(
        "--no-label",
        action="store_true",
        help="skip LLM labelling (fast topic-count tuning)",
    )
    ap.add_argument(
        "--refit", action="store_true", help="force a fresh fit even when assigning all"
    )
    args = ap.parse_args()

    reuse = (
        args.assign == "all" and not args.refit and CENTROIDS_OUT.exists() and TOPICS_OUT.exists()
    )
    if reuse:
        run_assign_all()
    else:
        run_fit(args)


if __name__ == "__main__":
    main()
