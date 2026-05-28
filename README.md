# TopicDrift

CS4530 Project 1 — analysing how research themes evolve at long-running software-engineering conferences. Currently scoped to ICSE (1976–2025); ICSA/ECSA next.

## Pipeline

Three stages, one folder under `src/` each:

```
src/ingest/         data team — fetch DBLP, enrich with OpenAlex, build preview CSV
src/analysis/       analysis team — TF-IDF, topic clustering, drift metrics
src/visualization/  viz team — figures for the paper
```

Data flow:

```
DBLP + OpenAlex APIs
        ↓
data/raw/            bronze — cached JSON responses
        ↓
data/interim/        silver — one row per paper, ML-ready
        ↓
data/gold/           gold — analysis-team aggregations (timelines, top terms, clusters)
        ↓
outputs/figures/     viz team plots
outputs/tables/      human-readable previews
```

## Run

```bash
pip install -r requirements.txt
make all          # ingest → clean → enrich → report
make clean-cache  # wipe data/ and outputs/ to force a full rebuild
```

Final output for humans: `outputs/tables/icse_papers_preview.csv` (~7,800 papers, 86 % with abstracts).

## Silver schema — the handoff to analysis

`data/interim/icse_enriched.parquet`. Don't change this schema without coordinating with the analysis team.

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

ACM Digital Library has no open bulk-metadata API. OpenAlex indexes ACM content and is the practical substitute.
