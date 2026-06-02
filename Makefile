PYTHON ?= python

.PHONY: all install ingest clean enrich enrich-titles enrich-acm enrich-full report topics yearly-topics clean-cache dump conf-report conf-scan

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

# analysis
topics:
	$(PYTHON) src/analysis/topics.py

yearly-topics:
	$(PYTHON) src/analysis/yearly_topics.py

clean-cache:
	rm -rf data outputs
