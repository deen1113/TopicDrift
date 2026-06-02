"""
topics.py — BERTopic topic-drift model over a pooled multi-conference corpus.

Corpus: all DBLP conferences with >= MIN_PAPERS papers and > MIN_HIT_RATE
abstract hit rate (from outputs/tables/conf_abstract_hit_rate_all.csv).
Input is data/interim/conf_enriched.parquet (built by build_conf_corpus.py).

Pipeline:
  embed (all-MiniLM-L6-v2)
  → UMAP (n_components=5, cosine)
  → HDBSCAN (min_cluster_size=15)
  → c-TF-IDF with stopwords from config/stopwords.txt + sklearn English
  → reduce_outliers(strategy="c-tf-idf")

Validation:
  1. Sanity events — known CS inflections recover their expected peak years
  2. Top papers per topic — most-recent 10 per topic for eyeballing
  3. Outlier timeline — share of -1 per year
  4. Stability (Adjusted Rand Index) — fit twice with different seeds
  5. Topic diversity — unique top-N words / total top-N slots
  6. LLM-as-judge — Claude rates each topic for coherence (skipped if no key)

Per-document fit score is stored in conf_paper_topics.parquet as
`topic_probability` — the model's max-class probability for that paper.

Writes:
  data/processed/conf_topics.parquet
  data/processed/conf_paper_topics.parquet
  data/processed/conf_topics_over_time.parquet
  outputs/tables/topic_sanity_events.csv
  outputs/tables/topic_top_papers.csv
  outputs/tables/topic_outlier_timeline.csv
  outputs/tables/topic_diversity_overlap.csv
  outputs/tables/topic_llm_ratings.csv   (if ANTHROPIC_API_KEY is set)
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

# Conference selection thresholds (applied to CONF_HIT_RATE_CSV at load time).
CONF_HIT_RATE_CSV = OUTPUTS_TABLES / "conf_abstract_hit_rate_all.csv"
MIN_PAPERS = 1000    # minimum papers in a conference to qualify
MIN_HIT_RATE = 0.95  # minimum abstract hit rate to qualify


def bucket_year(y: int) -> int:
    return (y // BUCKET_YEARS) * BUCKET_YEARS


def load_selected_corpus() -> pd.DataFrame:
    """Load papers from conferences meeting MIN_PAPERS and MIN_HIT_RATE thresholds."""
    ranks = pd.read_csv(CONF_HIT_RATE_CSV)
    keep = set(ranks[(ranks["n_total"] >= MIN_PAPERS) &
                     (ranks["abstract_hit_rate"] > MIN_HIT_RATE)]["conf"])
    df = pd.read_parquet(INTERIM_DIR / "conf_enriched.parquet")
    df = df[df["conf"].isin(keep) & df["has_abstract"]].reset_index(drop=True)
    print(f"Loaded {len(df):,} papers from {len(keep):,} conferences "
          f"(>= {MIN_PAPERS} papers, > {MIN_HIT_RATE:.0%} hit rate)")
    return df


def fit():
    df = load_selected_corpus()
    stopwords = load_stopwords()
    print(f"Fitting on {len(df):,} papers | {len(stopwords)} stopwords")

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
    pt.to_parquet(PROCESSED_DIR / "conf_paper_topics.parquet", index=False)
    print(f"  wrote conf_paper_topics.parquet ({len(pt)} rows)")

    info = tm.topic_info()
    info = tm.label_topics_llm(info, pt, df)
    keep = [c for c in ["topic_id", "size", "label", "llm_label", "top_words"] if c in info.columns]
    info[keep].to_parquet(PROCESSED_DIR / "conf_topics.parquet", index=False)
    print(f"  wrote conf_topics.parquet ({len(info)} rows)")

    tot = tm.topics_over_time(docs, timestamps)
    tot.to_parquet(PROCESSED_DIR / "conf_topics_over_time.parquet", index=False)
    print(f"  wrote conf_topics_over_time.parquet ({len(tot)} rows)")


if __name__ == "__main__":
    fit()
