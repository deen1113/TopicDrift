PYTHON ?= python
VENUE  ?= icse

.DEFAULT_GOAL := help

.PHONY: help install \
        dump scan rebuild-corpus conf-acm \
        venue titles acm venue-deep \
        corpus topics topics-all groups apply figures site analysis \
        human-review validate \
        zenodo status clean-data

SCOPES          := icse top10 all
FIGURE_INPUTS   := data/processed/conf_paper_topics.parquet \
                   data/processed/conf_topics.parquet \
                   data/processed/conf_topic_groups.parquet
STREAMGRAPH_OUT := $(foreach s,$(SCOPES),outputs/figures/topic_group_streamgraph_$(s).html)
TREEMAP_OUT     := $(foreach s,$(SCOPES),outputs/figures/topic_treemap_$(s).html)

$(STREAMGRAPH_OUT): $(FIGURE_INPUTS)
	$(PYTHON) -m topicdrift.visualization.topic_group_streamgraph

$(TREEMAP_OUT): $(FIGURE_INPUTS)
	$(PYTHON) -m topicdrift.visualization.topic_treemap

help:  ## Show this help
	@printf "\nTopicDrift — pipeline targets\n\n"
	@printf "Setup:\n"
	@awk 'BEGIN{FS=":.*?## "} /^(install):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nData — run once; both workflows read from these caches:\n"
	@awk 'BEGIN{FS=":.*?## "} /^(dump|scan):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nWorkflow A — one or more venues (preview CSV).  Requires: make dump\n"
	@printf "  Override venues with VENUE=\"icse ase issta\" (space-separated). Default: icse.\n"
	@printf "  Main-track filter is on by default; add INCLUDE_COMPANION=1 to keep companion/workshops too.\n"
	@awk 'BEGIN{FS=":.*?## "} /^(venue|titles|acm|venue-deep):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nWorkflow B — multi-conference topic drift (icse / top10 / all).  Requires: make scan\n"
	@printf "  Scopes live in config/venues.yaml. Each figure target writes one HTML per scope.\n"
	@printf "  Optional enrichment bridge (run before corpus if a venue has ACM-hosted papers):\n"
	@printf "    make venue-deep VENUE=icse   # full per-venue enrichment (OpenAlex + ACM scrape)\n"
	@printf "    make conf-acm   VENUE=icse   # ACM-scrape only + rebuild conf_enriched (skip if venue-deep already ran)\n"
	@awk 'BEGIN{FS=":.*?## "} /^(rebuild-corpus|conf-acm|corpus|topics|topics-all|groups|apply|figures|site|analysis):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nValidation (paper accuracy / coherence checks):\n"
	@awk 'BEGIN{FS=":.*?## "} /^(human-review|validate):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@printf "\nExtras:\n"
	@awk 'BEGIN{FS=":.*?## "} /^(zenodo|status|clean-data):.*?## /{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
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

rebuild-corpus:  ## Rebuild conf_enriched from scan cache + any per-venue parquets (no re-scan)
	$(PYTHON) -m topicdrift.ingest.build_conf_corpus

conf-acm:  ## ACM-scrape VENUE into <venue>_enriched, then rebuild conf_enriched (needs cookies)
	$(PYTHON) -m topicdrift.ingest.enrich_acm $(VENUE)
	$(PYTHON) -m topicdrift.ingest.build_conf_corpus

# ───────── Workflow A — single or multiple venues ─────────
# VENUE accepts one or more space-separated DBLP keys (e.g. icse, ase, issta).
# Each step iterates over the list. Requires `make dump` to have run.

venue:  ## Slice DBLP dump → OpenAlex enrich → preview CSV (main-track only by default)
	$(PYTHON) -m topicdrift.ingest.venue $(if $(INCLUDE_COMPANION),--include-companion) $(VENUE)
	$(PYTHON) -m topicdrift.ingest.enrich_openalex $(VENUE)
	$(PYTHON) -m topicdrift.ingest.report $(VENUE)

titles:  ## OpenAlex title-pass: recover abstracts for DOI-less rows (after `make venue`)
	$(PYTHON) -m topicdrift.ingest.enrich_openalex --title-pass $(VENUE)

acm:  ## ACM DL scrape: recover remaining abstracts (after `make venue`, needs cookie auth)
	$(PYTHON) -m topicdrift.ingest.enrich_acm $(VENUE)

venue-deep: venue titles acm  ## venue + title-pass + ACM scrape (full enrichment)

# ───────── Workflow B — multi-conference topic drift ──────
# Run order for a first-time setup (after `make scan`):
#   make corpus     # fast (no API calls)
#   make topics     # fits BERTopic on the stratified sample
#   make groups     # writes config/topic_groups.conf.yaml (edit between if desired)
#   make apply
#   make figures    # or `make site` to also copy HTML into docs/visualizations/
# Or: `make analysis` runs every step end-to-end.

corpus:  ## Stratified fit sample (skips if data/processed/conf_universe.parquet exists)
	@if [ -f data/processed/conf_universe.parquet ]; then \
	  echo "  ✓ data/processed/conf_universe.parquet exists — skipping (delete to force re-run)"; \
	else \
	  $(PYTHON) -m topicdrift.analysis.select_corpus; \
	fi

topics:  ## Fit BERTopic on the sample (skips if data/processed/conf_topics.parquet exists)
	@if [ -f data/processed/conf_topics.parquet ]; then \
	  echo "  ✓ data/processed/conf_topics.parquet exists — skipping (delete to force re-fit)"; \
	else \
	  $(PYTHON) -m topicdrift.analysis.topics_conf --assign sample; \
	fi

topics-all:  ## Extend topic assignment to every paper (slow; resumable)
	$(PYTHON) -m topicdrift.analysis.topics_conf --assign all

groups:  ## Map topics → curated themes (edit config/topic_groups.conf.yaml)
	$(PYTHON) -m topicdrift.analysis.map_seed_themes

apply:  ## Stamp groupings into conf_* tables + theme registry
	$(PYTHON) -m topicdrift.analysis.apply_topic_groups --prefix conf_ \
		--config config/topic_groups.conf.yaml --title "All Conferences"

figures: $(STREAMGRAPH_OUT) $(TREEMAP_OUT)  ## Render per-scope figures (skips if up to date)

site: figures  ## figures + copy HTML into docs/visualizations/ for the static site
	cp outputs/figures/topic_group_streamgraph_*.html docs/visualizations/
	cp outputs/figures/topic_treemap_*.html docs/visualizations/

analysis: corpus topics groups apply site  ## End-to-end: corpus → topics → groups → apply → site

CONF ?=

human-review:  ## Interactive annotation session (CONF=conf/icse to restrict venue)
	$(PYTHON) -m topicdrift.analysis.human_review $(if $(CONF),--conf $(CONF))

validate:  ## Validation tasks A–D → data/validation/ + outputs/figures/val_*
	$(PYTHON) -m topicdrift.analysis.validation

# ───────── Zenodo deposit ─────────────────────────────────
# Packages the pipeline outputs that reviewers need to reproduce figures without
# re-running the full pipeline. Raw caches (DBLP dump, OpenAlex scan) are
# excluded — they are re-downloadable from source and would bloat the deposit.

ZENODO_STAGING := dist/topicdrift-dataset
ZENODO_ZIP     := dist/topicdrift-dataset.zip

zenodo: site  ## Package processed data + figures → dist/topicdrift-dataset.zip
	@rm -rf $(ZENODO_STAGING)
	@mkdir -p $(ZENODO_STAGING)/data/processed \
	           $(ZENODO_STAGING)/figures \
	           $(ZENODO_STAGING)/config
	@cp data/processed/*.parquet   $(ZENODO_STAGING)/data/processed/
	@cp outputs/figures/*.html     $(ZENODO_STAGING)/figures/
	@cp config/topic_groups.conf.yaml $(ZENODO_STAGING)/config/
	@cp README.md                  $(ZENODO_STAGING)/
	@cd dist && zip -r topicdrift-dataset.zip topicdrift-dataset/
	@rm -rf $(ZENODO_STAGING)
	@printf "  ✓ $(ZENODO_ZIP)  "
	@du -sh $(ZENODO_ZIP) | awk '{printf "(size: %s)\n", $$1}'
	@printf "  Upload at: https://zenodo.org/deposit/new\n"

# ───────── Extras / utilities ─────────────────────────────

status:  ## Report which pipeline artifacts exist on disk
	@printf "── Shared data ──\n"
	@for f in data/interim/dblp_conf.parquet \
	          data/interim/conf_enriched.parquet ; do \
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
	@ls -1 outputs/figures/topic_group_streamgraph_*.html outputs/figures/topic_treemap_*.html 2>/dev/null | sed 's/^/  /' || echo "  (no figures)"

clean-data:  ## Wipe data/ and outputs/ — confirms first (loses cached API responses)
	@printf "Wipe data/ and outputs/? This deletes cached DBLP+OpenAlex responses. [y/N] " ; \
	 read ans ; [ "$$ans" = "y" ] || { echo "aborted." ; exit 1 ; }
	rm -rf data outputs
