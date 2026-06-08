"""
build_themes.py — make the 10 ICSE themes the unit of analysis, taking each
paper's theme straight from the global conf fit's grouping.

The retired 74-topic ICSE fit is gone. Theme membership is no longer guessed
here: `conf_paper_topics.parquet` already carries a `group` column (the 10 themes
from config/topic_groups.conf.yaml, assigned by the conf pipeline), so this
script just:

  1. takes the ICSE slice of conf_paper_topics and reads each paper's `group`,
  2. restricts to the project's curated ~6.6k-paper set (the keys of the prior
     icse_paper_topics fit) so citation/author enrichment stays available,
  3. renumbers the 10 themes as topic_id 0-9 (largest first), and
  4. rewrites the three tables the figures read, now keyed by theme:
       data/processed/icse_topics.parquet            (10 rows)
       data/processed/icse_paper_topics.parquet      (~6.6k papers)
       data/processed/icse_topics_over_time.parquet  (theme × 5-year window)

The per-theme defining words (overall, and per 5-year window for the lexical
view) are recomputed from the papers' own text with class-based TF-IDF (c-TF-IDF).

Reads:  data/processed/conf_paper_topics.parquet (group = theme),
        data/processed/icse_paper_topics.74topics.bak (defines the paper set),
        data/interim/icse_enriched.parquet (text)
Writes: the three icse_* tables above (originals kept as *.74topics.bak)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

STOPWORDS = Path("config/stopwords.txt")
PROCESSED = Path("data/processed")
INTERIM = Path("data/interim")

CONF_PT = PROCESSED / "conf_paper_topics.parquet"
SCOPE_REF = PROCESSED / "icse_paper_topics.74topics.bak"  # the curated 6.6k paper set

BUCKET = 5          # 5-year windows
TOP_OVERALL = 10    # defining words stored per theme
TOP_WINDOW = 5      # defining words per theme × window (lexical turnover)


def stopwords() -> list[str]:
    custom = [
        ln.strip().lower()
        for ln in STOPWORDS.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    return sorted(ENGLISH_STOP_WORDS.union(custom))


def ctfidf(counts: np.ndarray, vocab: np.ndarray, k: int) -> list[list[str]]:
    """Top-k class-distinctive terms per row of a (class × term) count matrix,
    using BERTopic's c-TF-IDF weighting: tf within the class × log(1 + Ā / fₜ)."""
    counts = counts.astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    tf = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    f_t = counts.sum(axis=0)
    avg = counts.sum() / max(counts.shape[0], 1)
    idf = np.log1p(avg / np.where(f_t == 0, 1, f_t))
    w = tf * idf
    out = []
    for row in w:
        top = np.argsort(row)[::-1]
        out.append([vocab[i] for i in top[:k] if row[i] > 0])
    return out


def class_counts(texts: pd.Series, classes: pd.Series, vec: CountVectorizer):
    """Fit `vec` on `texts`, then sum term counts per class label.
    Returns (class_labels, count_matrix[class × term], vocabulary)."""
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    labels = sorted(classes.unique())
    idx = {c: i for i, c in enumerate(labels)}
    mat = np.zeros((len(labels), X.shape[1]))
    rows = classes.map(idx).to_numpy()
    for r in range(X.shape[0]):
        mat[rows[r]] += X[r].toarray()[0]
    return labels, mat, vocab


def main() -> None:
    # ── paper → theme, straight from the conf grouping, on the curated set ───
    conf = pd.read_parquet(CONF_PT, columns=["dblp_key", "year", "group"])
    icse = conf[conf["dblp_key"].str.startswith("conf/icse/")].drop_duplicates("dblp_key")
    scope = set(pd.read_parquet(SCOPE_REF, columns=["dblp_key"])["dblp_key"])
    icse = icse[icse["dblp_key"].isin(scope) & icse["group"].notna()].copy()
    icse = icse.rename(columns={"group": "theme"})

    en = pd.read_parquet(INTERIM / "icse_enriched.parquet", columns=["dblp_key", "text"])
    df = icse.merge(en, on="dblp_key", how="left")
    df["text"] = df["text"].fillna("")
    df["bucket"] = (df["year"] // BUCKET * BUCKET).astype(int)

    sizes = df["theme"].value_counts()
    theme_order = sizes.index.tolist()              # largest first
    theme_id = {t: i for i, t in enumerate(theme_order)}

    sw = stopwords()
    token = r"(?u)\b[a-z][a-z]+\b"                   # alphabetic words only

    # overall vocabulary per theme
    vec = CountVectorizer(stop_words=sw, token_pattern=token, min_df=3)
    labels, mat, vocab = class_counts(df["text"], df["theme"], vec)
    overall = dict(zip(labels, ctfidf(mat, vocab, TOP_OVERALL)))

    # vocabulary per theme × window
    df["cls"] = df["theme"] + "||" + df["bucket"].astype(str)
    vecw = CountVectorizer(stop_words=sw, token_pattern=token, min_df=2)
    wlabels, wmat, wvocab = class_counts(df["text"], df["cls"], vecw)
    win = dict(zip(wlabels, ctfidf(wmat, wvocab, TOP_WINDOW)))

    # ── 1) icse_topics.parquet ───────────────────────────────────────────────
    topics = pd.DataFrame([
        {"topic_id": theme_id[t], "size": int(sizes[t]), "label": t,
         "top_words": overall.get(t, [])}
        for t in theme_order
    ]).sort_values("topic_id").reset_index(drop=True)

    # ── 2) icse_paper_topics.parquet (conf fit has no per-paper probability) ──
    paper_topics = pd.DataFrame({
        "dblp_key": df["dblp_key"],
        "year": df["year"].astype(int),
        "topic_id": df["theme"].map(theme_id).astype(int),
        "topic_probability": 1.0,
    }).reset_index(drop=True)

    # ── 3) icse_topics_over_time.parquet ──────────────────────────────────────
    bucket_totals = df.groupby("bucket").size()
    tot = (pd.DataFrame([
        {"topic_id": theme_id[theme], "top_words": ", ".join(win.get(f"{theme}||{bucket}", [])),
         "freq": int(len(g)), "year_bucket": int(bucket),
         "share": round(len(g) / int(bucket_totals[bucket]), 6)}
        for (theme, bucket), g in df.groupby(["theme", "bucket"])
    ]).sort_values(["topic_id", "year_bucket"]).reset_index(drop=True))

    # ── back up originals (once), then write ─────────────────────────────────
    for fname, out in {"icse_topics.parquet": topics,
                       "icse_paper_topics.parquet": paper_topics,
                       "icse_topics_over_time.parquet": tot}.items():
        path = PROCESSED / fname
        bak = path.with_suffix(".74topics.bak")
        if path.exists() and not bak.exists():
            shutil.copy2(path, bak)
        out.to_parquet(path, index=False)

    print(f"Themed {len(paper_topics)} ICSE papers into {len(theme_order)} themes "
          f"(source: conf_paper_topics.group).\n")
    print(f"{'#':>2}  {'theme':<40} {'papers':>6}  top words")
    for r in topics.itertuples():
        print(f"{r.topic_id:>2}  {r.label:<40} {r.size:>6}  {', '.join(r.top_words[:6])}")


if __name__ == "__main__":
    main()
