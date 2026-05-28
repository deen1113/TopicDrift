PYTHON ?= python

.PHONY: all ingest clean enrich report clean-cache

all: ingest clean enrich report

ingest:
	$(PYTHON) src/ingest/ingest.py

clean:
	$(PYTHON) src/ingest/clean.py

enrich:
	$(PYTHON) src/ingest/enrich_openalex.py

report:
	$(PYTHON) src/ingest/report.py

clean-cache:
	rm -rf data outputs
