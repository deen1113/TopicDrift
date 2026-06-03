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

## `analysis/`

Shared engine:
- **`topic_model.py`** — the reusable BERTopic wrapper (embed → UMAP → HDBSCAN →
  c-TF-IDF → outlier-merge → LLM labelling). Used by the fit below.

The site pipeline — runs in this order:
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
