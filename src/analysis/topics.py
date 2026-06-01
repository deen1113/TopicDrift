"""
topics.py — Global BERTopic fit over all ICSE papers.

Writes:
  data/processed/icse_topics.parquet
  data/processed/icse_paper_topics.parquet
  data/processed/icse_topics_over_time.parquet
"""
from pathlib import Path

import pandas as pd

from topic_model import TopicModel, load_stopwords

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
OUTPUTS_TABLES = Path("outputs/tables")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)

SEED = 42
BUCKET_YEARS = 5


def bucket_year(y: int) -> int:
    return (y // BUCKET_YEARS) * BUCKET_YEARS


def fit():
    df = pd.read_parquet(INTERIM_DIR / "icse_enriched.parquet")
    df = df[df["has_abstract"]].reset_index(drop=True)
    stopwords = load_stopwords()
    print(f"Fitting on {len(df)} papers | {len(stopwords)} stopwords")

    docs = df["text"].tolist()
    timestamps = df["year"].apply(bucket_year).tolist()

    print("\nComputing embeddings...")
    tm = TopicModel(seed=SEED, stopwords=stopwords)
    embeddings = tm.embed(docs)

    print("\n=== Primary fit (seed=42) ===")
    topics = tm.fit(docs, embeddings=embeddings)
    n_before = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"  → {n_before} topics")

    diversity_pre, _ = TopicModel.topic_diversity(tm.topic_info())
    groups = tm.merge_duplicates(docs)
    if groups:
        print(f"\n=== Merging duplicate topics (jaccard >= {tm.jaccard_merge_threshold}) ===")
        for g in groups:
            print(f"  merge {g}")
        topics = tm.topics_
        n_after = len(set(topics)) - (1 if -1 in topics else 0)
        diversity_post, _ = TopicModel.topic_diversity(tm.topic_info())
        print(f"  → {n_after} topics after merge (was {n_before}, "
              f"diversity {diversity_pre:.3f} → {diversity_post:.3f})")
    else:
        print(f"\n  No topic pairs above jaccard {tm.jaccard_merge_threshold}; nothing to merge")

    probabilities = tm.per_doc_probabilities(docs)

    pt = tm.assign_papers(df, probabilities=probabilities)
    pt.to_parquet(PROCESSED_DIR / "icse_paper_topics.parquet", index=False)
    print(f"  wrote icse_paper_topics.parquet ({len(pt)} rows)")

    info = tm.topic_info()
    info = tm.label_topics_llm(info, pt, df)
    keep = [c for c in ["topic_id", "size", "label", "llm_label", "top_words"] if c in info.columns]
    info[keep].to_parquet(PROCESSED_DIR / "icse_topics.parquet", index=False)
    print(f"  wrote icse_topics.parquet ({len(info)} rows)")

    tot = tm.topics_over_time(docs, timestamps)
    tot.to_parquet(PROCESSED_DIR / "icse_topics_over_time.parquet", index=False)
    print(f"  wrote icse_topics_over_time.parquet ({len(tot)} rows)")


if __name__ == "__main__":
    fit()
