"""
yearly_topics.py — Per-year BERTopic fits + centroid-based topic lineage matching.

Step 1 (per-year fits)
  For each year with ≥ MIN_PAPERS_PER_YEAR abstracts, fit a fresh TopicModel.
  Save (year, topic_id_local, size, top_words) and the L2-normalised topic
  centroid (mean sentence-transformer embedding of papers in that topic).

Step 2 (lineage matching)
  Compute cosine similarity for every cross-year (year_a, topic_a) ×
  (year_b, topic_b) centroid pair. Build a graph with edges where
  cos ≥ COSINE_LINEAGE_THRESHOLD, run union-find → lineages. A lineage
  is "the same topic across years."

Step 3 (persistence stats)
  Print and write:
    - lineage-length histogram (# lineages spanning 1, 2, 3, ... years)
    - % of yearly topics that belong to a lineage of length ≥ 2
    - sample of long-lived lineages with their top words per year

Writes:
  data/processed/icse_topics_per_year.parquet
  data/processed/icse_topic_centroids.npy           (row-aligned to the parquet)
  data/processed/icse_topic_lineages.parquet        (year, topic_id_local, lineage_id)
  outputs/tables/topic_lineage_length_hist.csv
  outputs/tables/topic_lineage_examples.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from topic_model import TopicModel, load_stopwords

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
OUTPUTS_TABLES = Path("outputs/tables")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)

SEED = 42
MIN_PAPERS_PER_YEAR = 40           # skip years too small to cluster meaningfully
MIN_TOPIC_SIZE_FLOOR = 4           # smallest cluster size we accept in any year
MIN_TOPIC_SIZE_CEIL = 15           # don't go above the global default
COSINE_LINEAGE_THRESHOLD = 0.92    # τ — cross-year cosine sim to join into a lineage.
                                   # Picked from a threshold sweep on ICSE centroids: cross-year
                                   # cosine medians sit at ~0.65 because all topics share SE
                                   # vocabulary, so anything below ~0.90 collapses everything
                                   # into one giant lineage. 0.92 gives ~6 multi-year lineages.


def _min_topic_size(n_papers: int) -> int:
    """Scale min_topic_size with year size so small years still produce topics."""
    return max(MIN_TOPIC_SIZE_FLOOR, min(MIN_TOPIC_SIZE_CEIL, n_papers // 20))


def _topic_centroid(embeddings: np.ndarray, member_mask: np.ndarray) -> np.ndarray:
    """Mean embedding of papers in a topic, L2-normalised for cosine similarity."""
    v = embeddings[member_mask].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------- Step 1: per-year fits ----------
def fit_yearly(df: pd.DataFrame, embeddings: np.ndarray, stopwords: list[str]):
    rows = []
    centroids = []
    for year, sub in df.groupby("year"):
        n = len(sub)
        if n < MIN_PAPERS_PER_YEAR:
            print(f"  {year}: {n} papers — skipping (<{MIN_PAPERS_PER_YEAR})")
            continue
        mts = _min_topic_size(n)
        print(f"\n=== {year}: {n} papers, min_topic_size={mts} ===")

        sub_idx = sub.index.to_numpy()
        docs = sub["text"].tolist()
        sub_emb = embeddings[sub_idx]

        tm = TopicModel(seed=SEED, min_topic_size=mts, stopwords=stopwords)
        topics = tm.fit(docs, embeddings=sub_emb)
        tm.merge_duplicates(docs)
        topics = tm.topics_

        info = tm.topic_info()
        topics_arr = np.asarray(topics)
        for _, t in info.iterrows():
            tid = int(t["topic_id"])
            if tid == -1:
                continue
            mask = topics_arr == tid
            centroid = _topic_centroid(sub_emb, mask)
            rows.append({
                "year": int(year),
                "topic_id_local": tid,
                "size": int(mask.sum()),
                "top_words": list(t["top_words"]),
            })
            centroids.append(centroid)

    topics_df = pd.DataFrame(rows)
    centroids_arr = np.vstack(centroids) if centroids else np.zeros((0, embeddings.shape[1]))
    return topics_df, centroids_arr


# ---------- Step 2: lineage matching ----------
def build_lineages(topics_df: pd.DataFrame, centroids: np.ndarray,
                   threshold: float) -> pd.DataFrame:
    """Union-find topics across years whose centroid cosine sim ≥ threshold.

    Within-year edges are skipped (duplicate-merge already ran per year)."""
    n = len(topics_df)
    if n == 0:
        return topics_df.assign(lineage_id=pd.Series(dtype=int))

    # Cosine sim = dot product since centroids are unit-norm.
    sim = centroids @ centroids.T
    years = topics_df["year"].to_numpy()

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if years[i] == years[j]:
                continue
            if sim[i, j] >= threshold:
                union(i, j)
                edges += 1

    roots = [find(i) for i in range(n)]
    # Re-id lineages 0..k-1 in stable order
    root_to_id: dict[int, int] = {}
    lineage_ids = []
    for r in roots:
        if r not in root_to_id:
            root_to_id[r] = len(root_to_id)
        lineage_ids.append(root_to_id[r])

    print(f"\n  Cross-year edges above τ={threshold}: {edges}")
    print(f"  Yearly topics: {n}  →  Lineages: {len(root_to_id)}")
    return topics_df.assign(lineage_id=lineage_ids)


# ---------- Step 3: persistence stats ----------
def report_persistence(topics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lineage-length histogram + examples of long-lived lineages."""
    span = topics_df.groupby("lineage_id")["year"].agg(
        years_present=lambda s: sorted(set(s.tolist())),
        n_years=lambda s: s.nunique(),
        n_yearly_topics=lambda s: len(s),
    ).reset_index()

    hist = (
        span.groupby("n_years")
        .agg(n_lineages=("lineage_id", "size"))
        .reset_index()
        .sort_values("n_years")
    )
    hist["pct_of_lineages"] = (100 * hist["n_lineages"] / hist["n_lineages"].sum()).round(1)

    total_yearly_topics = len(topics_df)
    persistent_topics = topics_df.merge(
        span[span["n_years"] >= 2][["lineage_id"]], on="lineage_id"
    )
    pct_persistent = 100 * len(persistent_topics) / total_yearly_topics if total_yearly_topics else 0.0

    print("\n=== Lineage length histogram ===")
    print(hist.to_string(index=False))
    print(f"\n  Total yearly topics: {total_yearly_topics}")
    print(f"  In a lineage spanning ≥2 years: {len(persistent_topics)} ({pct_persistent:.1f}%)")
    print(f"  Total lineages: {len(span)}")
    print(f"  Single-year (one-off) topics: {(span['n_years'] == 1).sum()}")

    # Examples: top 10 longest-lived lineages
    long_lived = span.sort_values("n_years", ascending=False).head(10)
    example_rows = []
    for _, lin in long_lived.iterrows():
        members = topics_df[topics_df["lineage_id"] == lin["lineage_id"]].sort_values("year")
        for _, m in members.iterrows():
            example_rows.append({
                "lineage_id": int(lin["lineage_id"]),
                "n_years": int(lin["n_years"]),
                "year": int(m["year"]),
                "size": int(m["size"]),
                "top_words": ", ".join(list(m["top_words"])[:8]),
            })
    examples = pd.DataFrame(example_rows)

    print("\n=== Top 10 longest-lived lineages ===")
    if not examples.empty:
        for lid, group in examples.groupby("lineage_id", sort=False):
            n_y = int(group["n_years"].iloc[0])
            print(f"\n  Lineage L{lid}  ({n_y} years)")
            for _, r in group.iterrows():
                print(f"    [{r['year']} n={r['size']:>3}]  {r['top_words']}")

    return hist, examples


# ---------- Orchestration ----------
def run():
    df = pd.read_parquet(INTERIM_DIR / "icse_enriched.parquet")
    df = df[df["has_abstract"]].reset_index(drop=True)
    stopwords = load_stopwords()
    print(f"Yearly topic fits on {len(df)} papers | {len(stopwords)} stopwords")

    print("\nComputing embeddings (one pass, sliced per year)...")
    embedder = TopicModel(stopwords=stopwords)  # only to access .embed
    embeddings = embedder.embed(df["text"].tolist())

    topics_df, centroids = fit_yearly(df, embeddings, stopwords)
    topics_df.to_parquet(PROCESSED_DIR / "icse_topics_per_year.parquet", index=False)
    np.save(PROCESSED_DIR / "icse_topic_centroids.npy", centroids)
    print(f"\n  wrote icse_topics_per_year.parquet ({len(topics_df)} yearly topics)")
    print(f"  wrote icse_topic_centroids.npy ({centroids.shape})")

    topics_df = build_lineages(topics_df, centroids, threshold=COSINE_LINEAGE_THRESHOLD)
    topics_df[["year", "topic_id_local", "lineage_id"]].to_parquet(
        PROCESSED_DIR / "icse_topic_lineages.parquet", index=False
    )

    hist, examples = report_persistence(topics_df)
    hist.to_csv(OUTPUTS_TABLES / "topic_lineage_length_hist.csv", index=False)
    examples.to_csv(OUTPUTS_TABLES / "topic_lineage_examples.csv", index=False)


if __name__ == "__main__":
    run()
