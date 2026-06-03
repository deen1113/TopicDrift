# TopicDrift

How research themes evolve across long-running software-engineering conferences. Built on DBLP metadata enriched with OpenAlex abstracts and clustered with BERTopic.

```bash
make install
make help
```

## Two workflows

Pick one depending on what you want.

### Shared data (one-time)

Both workflows read from the same on-disk caches. Build them once:

```bash
make dump       # download + parse DBLP XML → dblp_conf.parquet (~30 min, ~1 GB)
make scan       # dump + OpenAlex abstract scan + pooled corpus (~days, resumable)
```

`scan` depends on `dump`, so a fresh checkout can go straight to `make scan` if you want both. Workflow A only needs `dump`; Workflow B needs `scan`.

### A. One or more venues → preview CSVs

Slice the DBLP dump down to one (or more) venues, enrich with OpenAlex, write a human-readable CSV per venue. Requires `make dump`.

```bash
make venue                                # default: VENUE=icse
make venue VENUE="icse ase issta msr"     # any space-separated DBLP keys
make venue-deep VENUE=icse                # adds slow title-pass + ACM DL scrape
make venue INCLUDE_COMPANION=1            # keep companion volumes + workshops
```

A venue-agnostic main-track filter runs by default: it infers each venue's canonical acronym from the DBLP `booktitle` distribution, then keeps rows whose booktitle starts with that acronym (including split volumes like `ICSE (2)` and co-located research tracks like `ICSE-SEIP`, `ICSE (NIER)`). Companion volumes, workshop summaries, and `*@ICSE`-style satellites are dropped. The same rule works for renamed venues — e.g. `conf/kbse` correctly resolves to `ASE` and keeps both eras.

Outputs: `outputs/tables/<venue>_papers_preview.csv` per venue, plus silver-layer parquet at `data/interim/<venue>_enriched.parquet`.

### B. Multi-conference topic drift — icse / top10 / all

One global topic space fit on a stratified sample across every qualifying DBLP conference. The website renders three scopes — a single venue (ICSE), a curated top 10, and the full set — as venue filters over that same model. Scope membership lives in `config/venues.yaml`; edit the lists there to change what each scope shows. Requires `make scan`.

```bash
make corpus     # stratified fit sample (fast, no API)
make topics     # fit BERTopic on the sample
make groups     # map topics → curated themes (edit config/topic_groups.conf.yaml)
make apply
make figures    # writes one HTML per scope into outputs/figures/
make site       # also copies the HTML into docs/visualizations/
# or just:
make analysis   # corpus → topics → groups → apply → site
```

Per-scope outputs land at `outputs/figures/topic_group_streamgraph_{icse,top10,all}.html` and `topic_treemap_{icse,top10,all}.html`.

## Inspecting state

```bash
make status     # lists which pipeline artifacts exist on disk
make help       # full target list with one-line descriptions
make clean-data # wipe data/ and outputs/ (asks first; loses cached API responses)
```

## Layout

```
topicdrift/ingest/         DBLP fetch, OpenAlex enrichment, ACM DL recovery
topicdrift/analysis/       corpus selection, topic fitting, theme mapping
topicdrift/visualization/  per-scope streamgraph + treemap figures
topicdrift/topic_model.py  shared BERTopic wrapper (embed → UMAP → HDBSCAN → label)
config/                    venues.yaml (scopes), topic_groups.conf.yaml, stopwords.txt
data/raw/                  cached API responses
data/interim/, data/processed/   per-venue and pooled parquet tables
outputs/figures/, outputs/tables/  HTML figures and human-readable previews
docs/                      static site served from this repo
```

## Silver schema (per-venue handoff to analysis)

`data/interim/<venue>_enriched.parquet`:

| Column | Type | Source |
|---|---|---|
| `dblp_id`, `dblp_key` | str | DBLP |
| `title`, `year`, `doi`, `authors`, `url`, `ee` | mixed | DBLP |
| `has_doi`, `venue` | bool, str | derived |
| `abstract`, `has_abstract`, `text` | str | OpenAlex (`text` = normalised title + abstract) |
| `oa_concepts` | list[str] | OpenAlex concepts (score ≥ 0.3) |
| `citation_count`, `openalex_id`, `oa_type` | mixed | OpenAlex |

## Data sources

| Source | What we use |
|---|---|
| DBLP | titles, authors, years, DOIs |
| OpenAlex | abstracts, concept tags, citation counts |

ACM Digital Library has no open bulk-metadata API. OpenAlex indexes ACM content and is the practical substitute; `venue-deep` adds an optional ACM DL scrape pass for hard-to-recover abstracts (cookie auth required).
