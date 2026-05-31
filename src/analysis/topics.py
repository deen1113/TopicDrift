"""
topics.py — BERTopic MVP for ICSE topic drift, with five validation passes.

Pipeline:
  embed (all-MiniLM-L6-v2)
  → UMAP (n_components=5, cosine)
  → HDBSCAN (min_cluster_size=15)
  → c-TF-IDF with stopwords from config/stopwords.txt + sklearn English
  → reduce_outliers(strategy="c-tf-idf")

Validation:
  1. Sanity events — known SE inflections recover their expected peak years
  2. Top papers per topic — most-recent 10 per topic for eyeballing
  3. Outlier timeline — share of -1 per year
  4. Stability (Adjusted Rand Index) — fit twice with different seeds
  5. Topic diversity — unique top-N words / total top-N slots
  6. LLM-as-judge — Claude rates each topic for coherence (skipped if no key)

Per-document fit score is stored in icse_paper_topics.parquet as
`topic_probability` — the model's max-class probability for that paper.

Writes:
  data/processed/icse_topics.parquet
  data/processed/icse_paper_topics.parquet  (now with topic_probability)
  data/processed/icse_topics_over_time.parquet
  outputs/tables/topic_sanity_events.csv
  outputs/tables/topic_top_papers.csv
  outputs/tables/topic_outlier_timeline.csv
  outputs/tables/topic_diversity_overlap.csv
  outputs/tables/topic_llm_ratings.csv   (if ANTHROPIC_API_KEY is set)
"""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
from umap import UMAP
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
OUTPUTS_TABLES = Path("outputs/tables")
CONFIG_DIR = Path("config")
LLM_CACHE = Path("data/raw/llm_topic_ratings")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
LLM_CACHE.mkdir(parents=True, exist_ok=True)

SEED = 42
MIN_TOPIC_SIZE = 15
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BUCKET_YEARS = 5
BUCKET_TOLERANCE = 5
JACCARD_MERGE_THRESHOLD = 0.15  # top-word overlap above this → merge the topics

# Latest Claude Haiku — fast, cheap, plenty for a 1-5 rating task.
LLM_MODEL = "claude-haiku-4-5-20251001"

SANITY_EVENTS = [
    ("Object-oriented design", ["object", "object-oriented", "class"], 1990),
    ("Software process / CMM", ["process", "maturity", "improvement"], 1995),
    ("Agile / Scrum",          ["agile", "scrum", "extreme"], 2001),
    ("Mining software repos",  ["mining", "repository", "commit"], 2004),
    ("Cloud / SaaS",           ["cloud", "saas"], 2008),
    ("Mobile / Android",       ["android", "mobile", "smartphone"], 2010),
    ("Deep learning / neural", ["deep learning", "neural network", "neural"], 2014),
    ("DevOps / continuous",    ["devops", "continuous deployment"], 2015),
    ("LLM / code generation",  ["llm", "language model", "transformer"], 2022),
]


def load_stopwords() -> list[str]:
    """Custom SE stopwords from config/stopwords.txt + sklearn's English list."""
    words: set[str] = set()
    path = CONFIG_DIR / "stopwords.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                words.add(line)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    words.update(ENGLISH_STOP_WORDS)
    return list(words)


def bucket_year(y: int) -> int:
    return (y // BUCKET_YEARS) * BUCKET_YEARS


def _build_model(stopwords: list[str], seed: int) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean", cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(stop_words=stopwords)
    return BERTopic(
        embedding_model=EMBEDDING_MODEL,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=True,
    )


def _fit_and_reduce(docs, embeddings, seed, stopwords):
    model = _build_model(stopwords, seed)
    raw_topics, _ = model.fit_transform(docs, embeddings=embeddings)
    n_out_raw = sum(1 for t in raw_topics if t == -1)
    print(f"  pre-reduce outliers: {n_out_raw}/{len(raw_topics)} ({100*n_out_raw/len(raw_topics):.1f}%)")

    new_topics = model.reduce_outliers(docs, raw_topics, strategy="c-tf-idf")
    model.update_topics(docs, topics=new_topics,
                        vectorizer_model=CountVectorizer(stop_words=stopwords))
    n_out_new = sum(1 for t in new_topics if t == -1)
    print(f"  post-reduce outliers: {n_out_new}/{len(new_topics)} ({100*n_out_new/len(new_topics):.1f}%)")
    return model, new_topics


def _per_doc_probabilities(model, docs) -> np.ndarray:
    """Per-doc max probability across topics — proxy for 'fit confidence'."""
    topic_distr, _ = model.approximate_distribution(docs)
    return np.asarray(topic_distr).max(axis=1)


def _save_artifacts(model, df, topics, probabilities, timestamps):
    info = model.get_topic_info().rename(columns={
        "Topic": "topic_id", "Count": "size",
        "Name": "label", "Representation": "top_words",
    })
    keep = [c for c in ["topic_id", "size", "label", "top_words"] if c in info.columns]
    info[keep].to_parquet(PROCESSED_DIR / "icse_topics.parquet", index=False)
    print(f"  wrote icse_topics.parquet ({len(info)} rows)")

    pt = pd.DataFrame({
        "dblp_key": df["dblp_key"].values,
        "year": df["year"].values,
        "topic_id": topics,
        "topic_probability": probabilities.round(4),
    })
    pt.to_parquet(PROCESSED_DIR / "icse_paper_topics.parquet", index=False)
    print(f"  wrote icse_paper_topics.parquet ({len(pt)} rows)")

    tot = model.topics_over_time(df["text"].tolist(), timestamps, nr_bins=None).rename(columns={
        "Topic": "topic_id", "Words": "top_words",
        "Frequency": "freq", "Timestamp": "year_bucket",
    })
    bucket_totals = tot.groupby("year_bucket")["freq"].transform("sum")
    tot["share"] = tot["freq"] / bucket_totals
    tot.to_parquet(PROCESSED_DIR / "icse_topics_over_time.parquet", index=False)
    print(f"  wrote icse_topics_over_time.parquet ({len(tot)} rows)")


def _to_word_list(value) -> list[str]:
    if value is None:
        return []
    return [str(w).lower() for w in value]


# ---------- Validation pass 1: sanity events ----------
def _sanity_events(topics: pd.DataFrame, tot: pd.DataFrame) -> pd.DataFrame:
    topics = topics.copy()
    topics["_words_lc"] = topics["top_words"].apply(_to_word_list)
    rows = []
    for label, keywords, expected in SANITY_EVENTS:
        matches = []
        for _, t in topics.iterrows():
            if t["topic_id"] == -1:
                continue
            joined = " ".join(t["_words_lc"])
            if any(kw.lower() in joined for kw in keywords):
                matches.append((t["topic_id"], int(t["size"]), t["top_words"]))
        if not matches:
            rows.append({"event": label, "expected_year": expected,
                         "matched_topic_id": None, "peak_year": None,
                         "delta_years": None, "peak_share_pct": None,
                         "status": "no matching topic"})
            continue
        matches.sort(key=lambda x: -x[1])
        tid, _tsize, _twords = matches[0]
        ts = tot[tot["topic_id"] == tid].sort_values("year_bucket")
        if ts.empty:
            rows.append({"event": label, "expected_year": expected,
                         "matched_topic_id": int(tid), "peak_year": None,
                         "delta_years": None, "peak_share_pct": None,
                         "status": "no time-series"})
            continue
        peak = ts.loc[ts["share"].idxmax()]
        peak_year = int(peak["year_bucket"])
        delta = peak_year - expected
        rows.append({"event": label, "expected_year": expected,
                     "matched_topic_id": int(tid), "peak_year": peak_year,
                     "delta_years": delta,
                     "peak_share_pct": round(100 * float(peak["share"]), 1),
                     "status": "ok" if abs(delta) <= BUCKET_TOLERANCE else f"off by {delta:+d}y"})
    return pd.DataFrame(rows)


# ---------- Duplicate-topic merging ----------
def _find_duplicate_groups(overlap_df: pd.DataFrame, threshold: float) -> list[list[int]]:
    """Union-find groups of topics whose top-10 Jaccard overlap >= threshold."""
    high = overlap_df[overlap_df["jaccard"] >= threshold]
    if high.empty:
        return []

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, r in high.iterrows():
        a, b = int(r["topic_a"]), int(r["topic_b"])
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    components: dict[int, set[int]] = {}
    for t in parent:
        components.setdefault(find(t), set()).add(t)
    return [sorted(g) for g in components.values() if len(g) >= 2]


# ---------- Validation pass 5: topic diversity ----------
def _topic_diversity(topics: pd.DataFrame, top_n: int = 10):
    """Diversity = unique top-N words / (n_topics × N). Plus the K most-overlapping topic pairs."""
    df = topics[topics["topic_id"] != -1].copy()
    word_sets = {
        int(r["topic_id"]): set(list(r["top_words"])[:top_n])
        for _, r in df.iterrows()
    }
    all_words = [w for ws in word_sets.values() for w in ws]
    diversity = len(set(all_words)) / len(all_words) if all_words else 0.0

    pairs = []
    ids = list(word_sets)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            inter = word_sets[a] & word_sets[b]
            union = word_sets[a] | word_sets[b]
            if union:
                jacc = len(inter) / len(union)
                if jacc > 0:
                    pairs.append({
                        "topic_a": a, "topic_b": b,
                        "shared": len(inter),
                        "jaccard": round(jacc, 3),
                        "shared_words": ", ".join(sorted(inter)),
                    })
    overlap_df = pd.DataFrame(pairs).sort_values("jaccard", ascending=False)
    return diversity, overlap_df


# ---------- Validation pass 6: LLM-as-judge ----------
def _llm_rate_topic(top_words: list[str], sample_titles: list[str], client) -> dict:
    prompt = f"""You are evaluating a topic from an unsupervised topic model run on ICSE (software-engineering conference) papers.

Top words for this topic:
{', '.join(top_words[:15])}

A random sample of 5 paper titles assigned to this topic:
{chr(10).join(f"- {t}" for t in sample_titles[:5])}

Rate the topic's coherence on a 1-5 scale:
  1 = incoherent (random words, papers unrelated to each other)
  3 = somewhat coherent (mixed themes but a discernible centre)
  5 = tightly coherent (single clear research theme, papers all fit)

Also provide a 4-6 word topic label.

Respond with ONLY a JSON object:
{{"rating": <int 1-5>, "label": "<short label>", "reason": "<1 sentence>"}}"""

    h = hashlib.sha1((LLM_MODEL + prompt).encode()).hexdigest()[:16]
    cache_path = LLM_CACHE / f"{h}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        result = json.loads(m.group(0)) if m else {
            "rating": None, "label": "PARSE_FAIL", "reason": text[:200],
        }
    cache_path.write_text(json.dumps(result))
    return result


def _run_llm_judge(topics: pd.DataFrame, pt: pd.DataFrame, silver: pd.DataFrame):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n--- LLM-as-judge skipped (set ANTHROPIC_API_KEY to enable) ---")
        return None
    try:
        import anthropic
    except ImportError:
        print("\n--- LLM-as-judge skipped (pip install anthropic) ---")
        return None

    client = anthropic.Anthropic()
    merged = pt.merge(silver[["dblp_key", "title"]], on="dblp_key", how="left")
    results = []
    print(f"\n=== LLM-as-judge ({LLM_MODEL}) ===")
    for _, t in topics[topics["topic_id"] != -1].iterrows():
        tid = int(t["topic_id"])
        members = merged[merged["topic_id"] == tid]
        sample = members.sample(min(5, len(members)), random_state=42)
        result = _llm_rate_topic(list(t["top_words"]), sample["title"].tolist(), client)
        result.update({"topic_id": tid, "size": int(t["size"]),
                       "top_words": ", ".join(list(t["top_words"])[:8])})
        results.append(result)
    return pd.DataFrame(results)


# ---------- Sanity-check display ----------
def _show_top_topics(topics: pd.DataFrame, pt: pd.DataFrame, silver: pd.DataFrame,
                     n_topics: int = 6, n_papers: int = 5):
    # pt already has `year`; only pull `title` from silver to avoid a merge collision.
    merged = pt.merge(silver[["dblp_key", "title"]], on="dblp_key", how="left")
    top = topics[topics["topic_id"] != -1].sort_values("size", ascending=False).head(n_topics)
    print(f"\n=== Top {n_topics} topics, {n_papers} random papers each ===")
    for _, t in top.iterrows():
        words = ", ".join(list(t["top_words"])[:8])
        print(f"\n  TOPIC #{int(t['topic_id'])} (n={int(t['size'])}) — {words}")
        members = merged[merged["topic_id"] == t["topic_id"]]
        sample = members.sample(min(n_papers, len(members)), random_state=42)
        sample = sample.sort_values("year", ascending=False)
        for _, p in sample.iterrows():
            title = (p["title"] or "")[:100]
            print(f"    [{p['year']}  fit={p['topic_probability']:.2f}]  {title}")


# ---------- Orchestration ----------
def fit():
    df = pd.read_parquet(INTERIM_DIR / "icse_enriched.parquet")
    df = df[df["has_abstract"]].reset_index(drop=True)
    stopwords = load_stopwords()
    print(f"Fitting on {len(df)} papers | {len(stopwords)} stopwords")

    docs = df["text"].tolist()
    timestamps = df["year"].apply(bucket_year).tolist()

    print("\nComputing embeddings (cached for both fits)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = np.asarray(embedder.encode(docs, show_progress_bar=True))

    print("\n=== Primary fit (seed=42) ===")
    model, topics = _fit_and_reduce(docs, embeddings, SEED, stopwords)
    n_before = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"  → {n_before} topics")
    # Capture pre-merge topic assignments for the stability ARI computation;
    # the seed=43 fit doesn't get a merge step, so a fair comparison uses
    # pre-merge clusters on both sides.
    topics_pre_merge = list(topics)

    # Auto-merge over-split duplicate topics. We compute Jaccard overlap on the
    # in-memory model's top-words, group above-threshold pairs by connected
    # components, then call BERTopic.merge_topics in one shot.
    info = model.get_topic_info().rename(columns={
        "Topic": "topic_id", "Representation": "top_words",
    })
    diversity_pre, overlap_pre = _topic_diversity(info)
    groups = _find_duplicate_groups(overlap_pre, JACCARD_MERGE_THRESHOLD)
    if groups:
        print(f"\n=== Merging duplicate topics (jaccard >= {JACCARD_MERGE_THRESHOLD}) ===")
        for g in groups:
            print(f"  merge {g}")
        model.merge_topics(docs, topics_to_merge=groups)
        topics = list(model.topics_)
        n_after = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"  → {n_after} topics after merge (was {n_before}, "
              f"diversity {diversity_pre:.3f} → ", end="")
        info_post = model.get_topic_info().rename(columns={
            "Topic": "topic_id", "Representation": "top_words",
        })
        diversity_post, _ = _topic_diversity(info_post)
        print(f"{diversity_post:.3f})")
    else:
        print(f"\n  No topic pairs above jaccard {JACCARD_MERGE_THRESHOLD}; nothing to merge")

    probabilities = _per_doc_probabilities(model, docs)
    _save_artifacts(model, df, topics, probabilities, timestamps)

    topics_df = pd.read_parquet(PROCESSED_DIR / "icse_topics.parquet")
    pt = pd.read_parquet(PROCESSED_DIR / "icse_paper_topics.parquet")
    tot = pd.read_parquet(PROCESSED_DIR / "icse_topics_over_time.parquet")

    # Validation pass 1: sanity events
    se = _sanity_events(topics_df, tot)
    se.to_csv(OUTPUTS_TABLES / "topic_sanity_events.csv", index=False)
    ok = (se["status"] == "ok").sum()
    print(f"\nSanity events: {ok}/{len(se)} recovered within ±{BUCKET_TOLERANCE}y")
    print(se[["event", "expected_year", "peak_year", "delta_years",
              "peak_share_pct", "status"]].to_string(index=False))

    # Pass 2: top papers per topic (pt already has year; only pull title+abstract from silver)
    merged = pt.merge(df[["dblp_key", "title", "abstract"]], on="dblp_key", how="left")
    merged = merged.sort_values(["topic_id", "year"], ascending=[True, False])
    merged.groupby("topic_id").head(10).reset_index(drop=True).to_csv(
        OUTPUTS_TABLES / "topic_top_papers.csv", index=False)

    # Pass 3: outlier timeline
    pt2 = pt.assign(is_outlier=(pt["topic_id"] == -1))
    ot = pt2.groupby("year").agg(papers=("dblp_key", "size"),
                                 outliers=("is_outlier", "sum"))
    ot["outlier_pct"] = (100 * ot["outliers"] / ot["papers"]).round(1)
    ot.reset_index().to_csv(OUTPUTS_TABLES / "topic_outlier_timeline.csv", index=False)

    # Pass 4: stability via ARI — compare pre-merge clusters on both sides so the
    # merge step on seed=42 doesn't artificially deflate the score.
    print("\n=== Stability fit (seed=43) ===")
    _, topics_b = _fit_and_reduce(docs, embeddings, SEED + 1, stopwords)
    ari = adjusted_rand_score(topics_pre_merge, topics_b)
    print(f"\n  Adjusted Rand Index (seed 42 vs 43, pre-merge): {ari:.3f}")
    print("  Reference: 1.0 identical, 0.5+ reliably useful, 0 random")

    # Pass 5: topic diversity
    diversity, overlap = _topic_diversity(topics_df, top_n=10)
    overlap.to_csv(OUTPUTS_TABLES / "topic_diversity_overlap.csv", index=False)
    print(f"\n=== Topic diversity ===")
    print(f"  Unique top-10 words / total slots: {diversity:.3f}")
    print(f"  ({len(overlap)} topic pairs share ≥1 top-10 word)")
    if len(overlap):
        print("\n  Most-overlapping topic pairs (top 5):")
        for _, r in overlap.head(5).iterrows():
            print(f"    #{r.topic_a}↔#{r.topic_b}  jacc={r.jaccard}  shared: {r.shared_words}")

    # Pass 6: LLM-as-judge
    llm_df = _run_llm_judge(topics_df, pt, df)
    if llm_df is not None:
        llm_df.to_csv(OUTPUTS_TABLES / "topic_llm_ratings.csv", index=False)
        print(f"\n  LLM ratings: mean={llm_df['rating'].mean():.2f}/5  "
              f"({(llm_df['rating'] >= 4).sum()}/{len(llm_df)} rated ≥4)")
        print("\n  Lowest-rated topics:")
        for _, r in llm_df.sort_values("rating").head(5).iterrows():
            print(f"    #{r.topic_id} (n={r['size']})  rating={r.rating}  '{r.label}' — {r.reason[:120]}")

    # Sanity-check display
    _show_top_topics(topics_df, pt, df, n_topics=6, n_papers=5)


if __name__ == "__main__":
    fit()
