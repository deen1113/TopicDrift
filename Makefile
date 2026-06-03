PYTHON ?= python
VENUE  ?= icse

.DEFAULT_GOAL := help

.PHONY: help install \
        dump scan \
        venue venue-deep \
        corpus topics topics-all groups apply figures site analysis \
        status clean-data

help:  ## Show this help
	@printf "\nTopicDrift — pipeline targets\n\n"
	@printf "Setup:\n"
	@awk 'BEGIN{FS=":.*?## "} /^(install):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nData — run once; both workflows read from these caches:\n"
	@awk 'BEGIN{FS=":.*?## "} /^(dump|scan):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nWorkflow A — one or more venues (preview CSV).  Requires: make dump\n"
	@printf "  Override venues with VENUE=\"icse ase issta\" (space-separated). Default: icse.\n"
	@awk 'BEGIN{FS=":.*?## "} /^(venue|venue-deep):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nWorkflow B — multi-conference topic drift (icse / top10 / all).  Requires: make scan\n"
	@printf "  Scopes live in config/venues.yaml. Each figure target writes one HTML per scope.\n"
	@awk 'BEGIN{FS=":.*?## "} /^(corpus|topics|topics-all|groups|apply|figures|site|analysis):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nExtras:\n"
	@awk 'BEGIN{FS=":.*?## "} /^(status|clean-data):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\n"

install:  ## Editable install (pip install -e .)
	$(PYTHON) -m pip install -e .

# ───────── Data — shared prerequisites ────────────────────
# `dump` produces data/interim/dblp_conf.parquet, the per-paper DBLP slice
# both workflows read. `scan` extends it with OpenAlex abstracts across every
# conference; only Workflow B needs that extension.

dump:  ## Download + parse DBLP XML → dblp_conf.parquet (~30 min, ~1 GB)
	$(PYTHON) -m topicdrift.ingest.ingest_dblp_dump

scan: dump  ## dump + OpenAlex abstract scan + pooled corpus (~days, resumable)
	$(PYTHON) -m topicdrift.ingest.conf_abstract_report
	$(PYTHON) -m topicdrift.ingest.build_conf_corpus

# ───────── Workflow A — single or multiple venues ─────────
# VENUE accepts one or more space-separated DBLP keys (e.g. icse, ase, issta).
# Each step iterates over the list. Requires `make dump` to have run.

venue:  ## Slice DBLP dump → OpenAlex enrich → preview CSV
	$(PYTHON) -m topicdrift.ingest.venue $(VENUE)
	$(PYTHON) -m topicdrift.ingest.enrich_openalex $(VENUE)
	$(PYTHON) -m topicdrift.ingest.report $(VENUE)

venue-deep: venue  ## venue + slow title-pass + ACM DL scrape (ACM needs cookie auth)
	$(PYTHON) -m topicdrift.ingest.enrich_openalex --title-pass $(VENUE)
	$(PYTHON) -m topicdrift.ingest.enrich_acm $(VENUE)

# ───────── Workflow B — multi-conference topic drift ──────
# Run order for a first-time setup (after `make scan`):
#   make corpus     # fast (no API calls)
#   make topics     # fits BERTopic on the stratified sample
#   make groups     # writes config/topic_groups.conf.yaml (edit between if desired)
#   make apply
#   make figures    # or `make site` to also copy HTML into docs/visualizations/
# Or: `make analysis` runs every step end-to-end.

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
	@printf "── Shared data ──\n"
	@for f in data/interim/dblp_conf.parquet \
	          data/processed/conf_enriched.parquet ; do \
	  if [ -f $$f ]; then echo "  ✓ $$f"; else echo "  ✗ $$f (missing)"; fi ; \
	done
	@printf "\n── Workflow A (per-venue) ──\n"
	@ls -1 data/interim/*_dblp.parquet 2>/dev/null | grep -v "/dblp_conf\.parquet$$" | sed 's/^/  /' || echo "  (no venue slices)"
	@ls -1 outputs/tables/*_papers_preview.csv 2>/dev/null | sed 's/^/  /' || echo "  (no preview CSVs)"
	@printf "\n── Workflow B (multi-conference) ──\n"
	@for f in data/processed/conf_universe.parquet \
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
