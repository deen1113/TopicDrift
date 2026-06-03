"""
topic_model.py — Reusable BERTopic pipeline for ICSE topic-drift analysis.

Wraps embed → UMAP → HDBSCAN → c-TF-IDF → outlier-reduce → duplicate-merge
behind a class so the same machinery can drive the global fit (topics.py) and
per-year fits (yearly_topics.py) without duplicating code.
"""
import hashlib
import re
from functools import lru_cache
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP


CONFIG_DIR = Path("config")
LLM_CACHE = Path("data/raw/llm_topic_ratings")
LLM_CACHE.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Local instruct LLM for offline topic labelling (no API key needed). Open/ungated,
# runs on Apple-Silicon MPS in fp16. Swap for Qwen2.5-1.5B-Instruct if RAM is tight.
DEFAULT_LABEL_LLM = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MIN_TOPIC_SIZE = 15
DEFAULT_JACCARD_MERGE_THRESHOLD = 0.15

# Domain-specific lemma overrides for SE shorthand WordNet doesn't know.
# (WordNet leaves "apps" alone because "app" isn't in its noun inventory.)
_LEMMA_OVERRIDES = {
    "apps": "app", "devs": "dev", "libs": "lib", "repos": "repo",
    "vms": "vm", "uis": "ui", "guis": "gui", "apis": "api",
    "ides": "ide", "llms": "llm", "dnns": "dnn", "cnns": "cnn",
    "rnns": "rnn", "asts": "ast",
}

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# Filler words we never want a label to start or end on.
_LABEL_CONNECTORS = {
    "in", "with", "for", "of", "on", "to", "from", "the", "a", "an",
    "into", "via", "using", "based", "and", "or", "at", "by",
}


def _ensure_wordnet():
    for corpus in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)


_ensure_wordnet()
_LEMMATIZER = WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def lemmatize_token(token: str) -> str:
    """Noun-pass then verb-pass lemmatization with SE shorthand overrides."""
    t = token.lower()
    if t in _LEMMA_OVERRIDES:
        return _LEMMA_OVERRIDES[t]
    n = _LEMMATIZER.lemmatize(t, pos="n")
    return _LEMMATIZER.lemmatize(n, pos="v")


def lemmatizing_tokenizer(text: str) -> list[str]:
    return [lemmatize_token(t) for t in _TOKEN_RE.findall(text)]


def load_stopwords() -> list[str]:
    """SE stopwords from config/stopwords.txt + sklearn English list, lemmatized."""
    words: set[str] = set()
    path = CONFIG_DIR / "stopwords.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                words.add(line)
    words.update(ENGLISH_STOP_WORDS)
    return sorted({lemmatize_token(w) for w in words})


def _clean_label(text: str, max_words: int = 3) -> str:
    """Normalise an LLM label to title-case, max 3 words, no connectors/punctuation."""
    label = text.strip().splitlines()[0].strip() if text.strip() else ""
    label = label.strip('"\'`').rstrip(".:;,")
    for prefix in ("Topic label:", "Label:", "Topic:", "Topic name:"):
        if label.lower().startswith(prefix.lower()):
            label = label[len(prefix):].strip()
    if " " not in label and len(label) > 14:
        label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    label = re.sub(r"\s*&\s*|\s+and\s+", " ", label, flags=re.IGNORECASE)
    words = re.sub(r"\s+", " ", label).strip().split()[:max_words]
    while words and words[-1].lower() in _LABEL_CONNECTORS:
        words.pop()
    while words and words[0].lower() in _LABEL_CONNECTORS:
        words.pop(0)
    return " ".join(w.capitalize() for w in words)


def make_vectorizer(stopwords: list[str]) -> CountVectorizer:
    return CountVectorizer(
        tokenizer=lemmatizing_tokenizer,
        token_pattern=None,
        stop_words=stopwords,
    )


class TopicModel:
    """BERTopic wrapper reused across global and per-year fits."""

    def __init__(
        self,
        seed: int = 42,
        min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        jaccard_merge_threshold: float = DEFAULT_JACCARD_MERGE_THRESHOLD,
        stopwords: list[str] | None = None,
        cluster_selection_method: str = "eom",
    ):
        self.seed = seed
        self.min_topic_size = min_topic_size
        self.embedding_model_name = embedding_model
        self.jaccard_merge_threshold = jaccard_merge_threshold
        self.stopwords = stopwords if stopwords is not None else load_stopwords()
        # "eom" picks a few broad clusters; "leaf" gives many fine-grained topics
        # (needed for large, diverse corpora where eom collapses to a handful).
        self.cluster_selection_method = cluster_selection_method
        self.model: BERTopic | None = None
        self.topics_: list[int] | None = None

    def build_model(self, seed: int | None = None) -> BERTopic:
        s = self.seed if seed is None else seed
        umap_model = UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0,
            metric="cosine", random_state=s,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            metric="euclidean", cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,
        )
        return BERTopic(
            embedding_model=self.embedding_model_name,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=make_vectorizer(self.stopwords),
            calculate_probabilities=False,
            verbose=True,
        )

    def embedder(self) -> SentenceTransformer:
        """Cached sentence-transformer (load once, reuse across batches)."""
        if getattr(self, "_embedder_obj", None) is None:
            self._embedder_obj = SentenceTransformer(self.embedding_model_name)
        return self._embedder_obj

    def embed(self, docs: list[str]) -> np.ndarray:
        return np.asarray(self.embedder().encode(docs, show_progress_bar=True))

    def fit(
        self,
        docs: list[str],
        embeddings: np.ndarray | None = None,
        seed: int | None = None,
        reduce_outliers: bool = True,
    ) -> list[int]:
        self.model = self.build_model(seed=seed)
        raw_topics, _ = self.model.fit_transform(docs, embeddings=embeddings)
        n_out_raw = sum(1 for t in raw_topics if t == -1)
        print(f"  pre-reduce outliers: {n_out_raw}/{len(raw_topics)} ({100*n_out_raw/len(raw_topics):.1f}%)")

        if reduce_outliers and n_out_raw > 0:
            new_topics = self.model.reduce_outliers(docs, raw_topics, strategy="c-tf-idf")
            self.model.update_topics(
                docs, topics=new_topics,
                vectorizer_model=make_vectorizer(self.stopwords),
            )
            n_out_new = sum(1 for t in new_topics if t == -1)
            print(f"  post-reduce outliers: {n_out_new}/{len(new_topics)} ({100*n_out_new/len(new_topics):.1f}%)")
            self.topics_ = list(new_topics)
        else:
            self.topics_ = list(raw_topics)
        return self.topics_

    def per_doc_probabilities(self, docs: list[str]) -> np.ndarray:
        """Per-doc max probability — proxy for fit confidence."""
        self._require_fit()
        topic_distr, _ = self.model.approximate_distribution(docs)
        return np.asarray(topic_distr).max(axis=1)

    def topic_info(self) -> pd.DataFrame:
        self._require_fit()
        return self.model.get_topic_info().rename(columns={
            "Topic": "topic_id", "Count": "size",
            "Name": "label", "Representation": "top_words",
        })

    def assign_papers(
        self,
        df: pd.DataFrame,
        probabilities: np.ndarray | None = None,
    ) -> pd.DataFrame:
        self._require_fit()
        if probabilities is None:
            probabilities = np.zeros(len(self.topics_))
        return pd.DataFrame({
            "dblp_key": df["dblp_key"].values,
            "year": df["year"].values,
            "topic_id": self.topics_,
            "topic_probability": np.asarray(probabilities).round(4),
        })

    def topics_over_time(self, docs: list[str], timestamps: list) -> pd.DataFrame:
        self._require_fit()
        tot = self.model.topics_over_time(docs, timestamps, nr_bins=None).rename(columns={
            "Topic": "topic_id", "Words": "top_words",
            "Frequency": "freq", "Timestamp": "year_bucket",
        })
        bucket_totals = tot.groupby("year_bucket")["freq"].transform("sum")
        tot["share"] = tot["freq"] / bucket_totals
        return tot

    def merge_duplicates(
        self,
        docs: list[str],
        threshold: float | None = None,
    ) -> list[list[int]]:
        """Auto-merge over-split topics by Jaccard top-word overlap. Returns groups merged."""
        self._require_fit()
        thresh = self.jaccard_merge_threshold if threshold is None else threshold
        _, overlap = self.topic_diversity(self.topic_info())
        groups = self.find_duplicate_groups(overlap, thresh)
        if groups:
            self.model.merge_topics(docs, topics_to_merge=groups)
            self.topics_ = list(self.model.topics_)
        return groups

    @staticmethod
    def topic_diversity(topics: pd.DataFrame, top_n: int = 10):
        """Diversity score + per-pair Jaccard overlap table. Used by merge_duplicates."""
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
        cols = ["topic_a", "topic_b", "shared", "jaccard", "shared_words"]
        overlap_df = pd.DataFrame(pairs, columns=cols)
        if not overlap_df.empty:
            overlap_df = overlap_df.sort_values("jaccard", ascending=False)
        return diversity, overlap_df

    @staticmethod
    def find_duplicate_groups(overlap_df: pd.DataFrame, threshold: float) -> list[list[int]]:
        """Union-find groups of topics whose top-word Jaccard >= threshold."""
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

    def label_topics_llm(
        self,
        topics_df: pd.DataFrame,
        paper_topics: pd.DataFrame,
        silver: pd.DataFrame,
        model_name: str = DEFAULT_LABEL_LLM,
        max_titles: int = 8,
    ) -> pd.DataFrame:
        """Generate a concise 2-3 word label per topic using a local instruct LLM.

        Runs fully offline on MPS/CUDA/CPU. Raw generations are cached under
        data/raw/llm_topic_ratings/ so re-runs only re-apply cleaning."""
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            print("transformers/torch not installed — skipping LLM labelling")
            return topics_df

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=== Local-LLM topic labelling ({model_name} on {device}) ===")
        gen = pipeline(
            "text-generation", model=model_name,
            dtype=torch.float16, device=device,
        )
        tok = gen.tokenizer

        merged = paper_topics.merge(silver[["dblp_key", "title"]], on="dblp_key", how="left")
        system = (
            "You name software-engineering research topics. Reply with ONLY a topic "
            "name of 2 or 3 plain words separated by single spaces. Use title case. "
            "Do not use ampersands, the word 'and', punctuation, or run-on words. "
            "Example reply: Automated Program Repair"
        )

        labels = {}
        for _, t in topics_df[topics_df["topic_id"] != -1].iterrows():
            tid = int(t["topic_id"])
            top_words = list(t["top_words"])
            words = ", ".join(top_words[:10])
            titles = merged[merged["topic_id"] == tid]["title"].dropna()
            sample = titles.sample(min(max_titles, len(titles)), random_state=42)
            title_block = "\n".join(f"- {s[:90]}" for s in sample.tolist())
            user = (f"Keywords: {words}\n\nExample paper titles:\n{title_block}\n\n"
                    "Topic name:")
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
            h = hashlib.sha1((model_name + prompt).encode()).hexdigest()[:16]
            cache_path = LLM_CACHE / f"label_{h}.txt"
            if cache_path.exists():
                raw = cache_path.read_text()
            else:
                out = gen(prompt, max_new_tokens=12, do_sample=False,
                          return_full_text=False, pad_token_id=tok.eos_token_id)
                raw = out[0]["generated_text"]
                cache_path.write_text(raw)
            label = _clean_label(raw)
            if not label:
                label = " ".join(w.capitalize() for w in top_words[:2])
            labels[tid] = label
            print(f"  #{tid:>2} — {label}")

        topics_df = topics_df.copy()
        topics_df["llm_label"] = topics_df["topic_id"].map(labels)
        return topics_df

    def _require_fit(self):
        if self.model is None:
            raise RuntimeError("TopicModel not fit yet — call .fit(docs) first.")
