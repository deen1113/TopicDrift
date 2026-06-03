# `src/` — how the code is organized

Data flows through three layers, one folder each. This is the whole reason for
the split:

```
ingest/         raw DBLP + OpenAlex/ACM   ->  data/interim, data/processed (parquet)
analysis/       parquet                    ->  topic tables (data/processed/*topics*.parquet)
visualization/  topic tables               ->  interactive HTML (outputs/figures, docs/visualizations)
```

- **ingest/** gets the papers and abstracts and writes tidy parquet. No modelling.
- **analysis/** turns papers into *topics* (BERTopic) and groups topics into
  *themes*. No plotting.
- **visualization/** reads the analysis outputs and writes the self-contained
  Plotly HTML that the website embeds. No modelling.

If you only care about the website, you live in **analysis/** + **visualization/**.

---

## ⚠️ Two topic pipelines write the same files — pick one

The project grew from *ICSE-only* to *all DBLP conferences*, which left **two**
multi-conference topic pipelines. They both write
`data/processed/conf_topics.parquet` and `conf_paper_topics.parquet`, so
**running both clobbers** — and their schemas differ.

| | `analysis/topics.py` (A) | `analysis/topics_conf.py` (B) |
|---|---|---|
| input | `data/interim/conf_enriched.parquet` (from `build_conf_corpus.py`) | `data/processed/conf_enriched.parquet` |
| venues kept | ≥1000 papers & >95% abstracts | ≥50 papers & ≥50% abstracts |
| fit | BERTopic over **all** selected papers | fit on a stratified sample, then assign every paper by nearest centroid |
| `conf_paper_topics` cols | `dblp_key, year, topic_id, topic_probability` | `dblp_key, conf, year, topic_id` (+ `group` after apply) |
| Make target | `make topics` | `make corpus conf-topics` |

**The live 3-tab website is built by pipeline (B).** The scope figures need the
`conf` and `group` columns that only (B) produces, so running `make topics` (A)
will break them. The team should decide which becomes canonical.

---

## `analysis/`

Shared engine:
- **`topic_model.py`** — the reusable BERTopic wrapper (embed → UMAP → HDBSCAN →
  c-TF-IDF → outlier-merge → LLM labelling). Used by every fit below.

ICSE-origin lineage:
- **`topics.py`** — pipeline (A) above; pooled multi-conference BERTopic fit.
- **`yearly_topics.py`** — per-year fits + centroid topic-lineage matching (the
  "same topic across years" analysis). Independent of the website.

Multi-conference site lineage (B) — runs in this order:
1. **`select_corpus.py`** — choose venues + build the stratified fit sample
   (`conf_universe.parquet`).
2. **`topics_conf.py`** — global fit on the sample + nearest-centroid assignment
   (`--assign sample|all`). Writes `conf_topics` + `conf_paper_topics` + centroids.
3. **`map_seed_themes.py`** — assign each topic to one of the 10 locked ICSE
   themes by nearest anchor. Writes `config/topic_groups.conf.yaml` (editable).
4. **`apply_topic_groups.py`** — stamp the `group` column + registry
   (`conf_topic_groups.parquet`) into the tables from that YAML.

## `visualization/`

Shared:
- **`_common.py`** — data loaders + scope helpers shared by the figures.

Every script here is embedded in the live site:
- **`topic_group_streamgraph.py`** → `topic_group_streamgraph_{icse,top10,all}.html`
  (theme drift over time; reads the global `conf_*` tables, one figure per scope).
- **`topic_scope_treemap.py`** → `topic_scope_treemap_{icse,top10,all}.html`
  (decade → theme → topic composition; per scope).
- **`topic_drift_search.py`** → `topic_drift_search.html` (keyword drift explorer;
  reads the ICSE `icse_*` tables).

---

## Run order (Makefile)

```bash
# build the multi-conference site (pipeline B):
make corpus conf-topics conf-group conf-apply conf-site
# then extend assignment to the full corpus and refresh:
make conf-assign-all conf-apply conf-site
```

`make conf-site` regenerates the scope figures and copies them into
`docs/visualizations/`.

## Open cleanups (need team agreement)

- Resolve the **(A) vs (B)** topic-pipeline overlap above — one canonical fit.
