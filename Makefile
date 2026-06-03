PYTHON ?= python
VENUE  ?= icse

.DEFAULT_GOAL := help

.PHONY: help install \
        venue venue-deep \
        scan corpus topics topics-all groups apply figures site analysis \
        status clean-data

help:  ## Show this help
	@printf "\nTopicDrift — pipeline targets\n\n"
	@printf "Setup:\n"
	@awk 'BEGIN{FS=":.*?## "} /^(install):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nWorkflow A — single or multiple venues (DBLP + OpenAlex → preview CSV)\n"
	@printf "  Override venues with VENUE=\"icse ase issta\" (space-separated). Default: icse.\n"
	@awk 'BEGIN{FS=":.*?## "} /^(venue|venue-deep):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nWorkflow B — multi-conference topic drift (icse / top10 / all scopes)\n"
	@printf "  Scopes live in config/venues.yaml. Each figure target writes one HTML per scope.\n"
	@awk 'BEGIN{FS=":.*?## "} /^(scan|corpus|topics|topics-all|groups|apply|figures|site|analysis):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nExtras:\n"
	@awk 'BEGIN{FS=":.*?## "} /^(status|clean-data):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\n"

install:  ## Editable install (pip install -e .)
	$(PYTHON) -m pip install -e .

# ───────── Workflow A — single or multiple venues ─────────
# VENUE accepts one or more space-separated DBLP keys (e.g. icse, ase, issta).
# Defaults to icse. Each ingest step iterates over the list.

venue:  ## DBLP fetch → clean → OpenAlex enrich → preview CSV
	$(PYTHON) -m topicdrift.ingest.ingest $(VENUE)
	$(PYTHON) -m topicdrift.ingest.clean $(VENUE)
	$(PYTHON) -m topicdrift.ingest.enrich_openalex $(VENUE)
	$(PYTHON) -m topicdrift.ingest.report $(VENUE)

venue-deep: venue  ## venue + slow title-pass + ACM DL scrape (ACM needs cookie auth)
	$(PYTHON) -m topicdrift.ingest.enrich_openalex --title-pass $(VENUE)
	$(PYTHON) -m topicdrift.ingest.enrich_acm $(VENUE)

# ───────── Workflow B — multi-conference topic drift ──────
# Run order for a first-time setup:
#   make scan       # slow (~days, paced by OpenAlex daily budget; resumable)
#   make corpus     # fast (no API calls)
#   make topics     # fits BERTopic on the stratified sample
#   make groups     # writes config/topic_groups.conf.yaml (edit between if desired)
#   make apply
#   make figures    # or `make site` to also copy HTML into docs/visualizations/
# Or: `make analysis` runs every step end-to-end.

scan:  ## DBLP-wide dump + OpenAlex scan + pooled corpus (~days, resumable)
	$(PYTHON) -m topicdrift.ingest.ingest_dblp_dump
	$(PYTHON) -m topicdrift.ingest.conf_abstract_report
	$(PYTHON) -m topicdrift.ingest.build_conf_corpus

corpus:  ## Stratified fit sample (data/processed/conf_universe.parquet)
	$(PYTHON) -m topicdrift.analysis.select_corpus

topics:  ## Fit BERTopic on the sample + assign sample papers (fast)
	$(PYTHON) -m topicdrift.analysis.topics_conf --assign sample

topics-all:  ## Extend topic assignment to every paper (slow; resumable)
	$(PYTHON) -m topicdrift.analysis.topics_conf --assign all

groups:  ## Map topics → curated themes (edit config/topic_groups.conf.yaml)
	$(PYTHON) -m topicdrift.analysis.map_seed_themes

apply:  ## Stamp groupings into conf_* tables + theme registry
	$(PYTHON) -m topicdrift.analysis.apply_topic_groups --prefix conf_ \
		--config config/topic_groups.conf.yaml --title "All Conferences"

figures:  ## Render per-scope figures (one HTML each for icse / top10 / all)
	$(PYTHON) -m topicdrift.visualization.topic_group_streamgraph
	$(PYTHON) -m topicdrift.visualization.topic_scope_treemap

site: figures  ## figures + copy HTML into docs/visualizations/ for the static site
	cp outputs/figures/topic_group_streamgraph_*.html docs/visualizations/
	cp outputs/figures/topic_scope_treemap_*.html docs/visualizations/

analysis: corpus topics groups apply site  ## End-to-end: corpus → topics → groups → apply → site

# ───────── Extras / utilities ─────────────────────────────

status:  ## Report which pipeline artifacts exist on disk
	@printf "── Workflow A (single venue) ──\n"
	@ls -1 data/interim/*_enriched.parquet 2>/dev/null || echo "  (no enriched venues)"
	@ls -1 outputs/tables/*_papers_preview.csv 2>/dev/null || echo "  (no preview CSVs)"
	@printf "\n── Workflow B (multi-conference) ──\n"
	@for f in data/interim/dblp_conf.parquet \
	          data/processed/conf_enriched.parquet \
	          data/processed/conf_universe.parquet \
	          data/processed/conf_topics.parquet \
	          data/processed/conf_paper_topics.parquet \
	          data/processed/conf_topic_groups.parquet ; do \
	  if [ -f $$f ]; then echo "  ✓ $$f"; else echo "  ✗ $$f (missing)"; fi ; \
	done
	@printf "\n── Figures ──\n"
	@ls -1 outputs/figures/topic_group_streamgraph_*.html outputs/figures/topic_scope_treemap_*.html 2>/dev/null | sed 's/^/  /' || echo "  (no figures)"

clean-data:  ## Wipe data/ and outputs/ — confirms first (loses cached API responses)
	@printf "Wipe data/ and outputs/? This deletes cached DBLP+OpenAlex responses. [y/N] " ; \
	 read ans ; [ "$$ans" = "y" ] || { echo "aborted." ; exit 1 ; }
	rm -rf data outputs
