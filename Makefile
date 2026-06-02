PYTHON ?= python

.PHONY: all full install ingest clean enrich enrich-titles enrich-acm enrich-full report topics yearly-topics \
        clean-cache dump conf-report conf-scan conf-corpus \
        corpus conf-topics conf-assign-all conf-group conf-apply conf-viz conf-site

# Standard pipeline (no ACM auth required, no title pass)
all: ingest clean enrich report

# Full pipeline including title pass and ACM DL scrape
full: ingest clean enrich enrich-titles enrich-acm report

install:
	$(PYTHON) -m pip install -r requirements.txt

# ingest pipeline
ingest:
	$(PYTHON) src/ingest/ingest.py

clean:
	$(PYTHON) src/ingest/clean.py

# DOI pass only
enrich:
	$(PYTHON) src/ingest/enrich_openalex.py

# Optional title search pass. Slow (1 req per DOI-less paper); use on small corpora only
enrich-titles:
	$(PYTHON) src/ingest/enrich_openalex.py --title-pass

# Optional: recover abstracts from ACM DL (cookie auth required). Slow (1 req per ACM paper); use on small corpora only
enrich-acm:
	$(PYTHON) src/ingest/enrich_acm.py

# Run DOI pass + ACM scrape
enrich-full: enrich enrich-titles enrich-acm

report:
	$(PYTHON) src/ingest/report.py

# DBLP-wide abstract hit-rate scan (all conferences)
dump:
	$(PYTHON) src/ingest/ingest_dblp_dump.py

conf-report:
	$(PYTHON) src/ingest/conf_abstract_report.py

conf-scan: dump conf-report

# Build pooled corpus from dump + scan cache (no API calls); prerequisite for `make topics`
conf-corpus:
	$(PYTHON) src/ingest/build_conf_corpus.py

# analysis
topics:
	$(PYTHON) src/analysis/topics.py

yearly-topics:
	$(PYTHON) src/analysis/yearly_topics.py

# ── Multi-conference global pipeline (data/processed/conf_enriched.parquet) ────
# One global topic space; the website tabs are venue filters over it.
# Typical first run:   make corpus conf-topics conf-group conf-apply conf-site
# Then, in the background, extend the assignment to every paper:
#                      make conf-assign-all conf-apply conf-site

# 1. pick venues + build the stratified fit sample
corpus:
	$(PYTHON) src/analysis/select_corpus.py

# 2. fit on the sample, label, assign the sample (fast, working website)
conf-topics:
	PYTHONPATH=src/analysis $(PYTHON) src/analysis/topics_conf.py --assign sample

# 2b. extend assignment to the whole 2.3M-paper universe (slow; resumable)
conf-assign-all:
	PYTHONPATH=src/analysis $(PYTHON) src/analysis/topics_conf.py --assign all

# 3. fit topics into the 10 ICSE themes (editable: config/topic_groups.conf.yaml)
conf-group:
	PYTHONPATH=src/analysis $(PYTHON) src/analysis/map_seed_themes.py

# 4. stamp the grouping into the conf_* tables + registry/report
conf-apply:
	$(PYTHON) src/analysis/apply_topic_groups.py --prefix conf_ \
		--config config/topic_groups.conf.yaml --title "All Conferences"

# 5. (re)generate the per-scope figures and copy them into docs/
conf-viz:
	PYTHONPATH=src/visualization $(PYTHON) src/visualization/topic_group_streamgraph.py
	PYTHONPATH=src/visualization $(PYTHON) src/visualization/topic_scope_treemap.py

conf-site: conf-viz
	cp outputs/figures/topic_group_streamgraph_*.html docs/visualizations/
	cp outputs/figures/topic_scope_treemap_*.html docs/visualizations/

clean-cache:
	rm -rf data outputs
