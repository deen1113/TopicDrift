# TopicDrift — Data Documentation

See more on data and its usage below.
Live site: https://deen1113.github.io/TopicDrift/

---

## 1. Datasets ingested

| Source | What we take | Access |
|---|---|---|
| **DBLP** | Titles, authors, years, DOIs, venue for all `conf/*` papers | Full XML dump, parsed locally |
| **OpenAlex** | Abstracts, concept tags, citation counts | REST API (cached); open substitute for ACM DL |
| **ACM DL** *(optional)* | Abstracts OpenAlex missed | Authenticated scrape, on demand |

**Result:** ~2.3M papers across 2,118 conferences. ICSE (headline venue): ~11,000 papers, 1976–2025, ~98% with abstracts.

## 2. Cleaning

- **Filter to research papers:** `conf/*` only; drop corrigenda/datasets; keep main-track only (drop companion volumes, workshops, panels, tutorials, keynotes).
- **Deduplicate:** by `dblp_key` and `title + year`.
- **Recover abstracts:** OpenAlex (by DOI) → ACM scrape → title-only fallback; `has_abstract` flag records which.
- **Normalize text:** model input = normalized `title + abstract`; SE-specific stopwords applied at topic modeling.

## 3. Location & reproducibility

Three derived tiers (`data/` is gitignored — it's fully regenerable):

```
data/raw/        cached raw API responses (bronze)
data/interim/    one row per paper + abstract/text (silver)
data/processed/  topic model outputs: topics, assignments, themes (gold)
```

**Reproduce:**
```bash
make scan && make analysis     # rebuild everything from source
make zenodo                    # OR package a ready-to-use snapshot
```

## 4. Accessing the data

Gold tables are plain Parquet:

```python
import pandas as pd

papers = pd.read_parquet("data/processed/conf_paper_topics.parquet")
# columns: dblp_key, conf, year, topic_id, group(theme)

icse = papers[papers.conf == "conf/icse"]
print(icse.groupby("group").dblp_key.nunique())   # ICSE papers per theme
```

| File (`data/processed/`) | Contents |
|---|---|
| `conf_topics.parquet` | 174 topics: label, keywords, size, theme |
| `conf_paper_topics.parquet` | one row per paper → topic + theme |
| `conf_topic_groups.parquet` | 10 themes: paper count, share |
| `conf_*_centroids.npy` / `conf_fit_emb.npy` | embeddings (for coherence checks) |
