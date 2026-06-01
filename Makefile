PYTHON ?= python

.PHONY: all ingest clean enrich report topics yearly-topics clean-cache

all: ingest clean enrich report

# ingest pipeline (data team)
ingest:
	$(PYTHON) src/ingest/ingest.py

clean:
	$(PYTHON) src/ingest/clean.py

enrich:
	$(PYTHON) src/ingest/enrich_openalex.py

report:
	$(PYTHON) src/ingest/report.py

# analysis (analysis team)
topics:
	$(PYTHON) src/analysis/topics.py

yearly-topics:
	$(PYTHON) src/analysis/yearly_topics.py

clean-cache:
	rm -rf data outputs
