PYTHON ?= python

.PHONY: all install ingest clean enrich report topics yearly-topics clean-cache dump conf-report conf-scan

all: ingest clean enrich report

install:
	$(PYTHON) -m pip install -r requirements.txt

# ingest pipeline (data team)
ingest:
	$(PYTHON) src/ingest/ingest.py

clean:
	$(PYTHON) src/ingest/clean.py

enrich:
	$(PYTHON) src/ingest/enrich_openalex.py

report:
	$(PYTHON) src/ingest/report.py

# DBLP-wide abstract hit-rate scan (all conferences)
dump:
	$(PYTHON) src/ingest/ingest_dblp_dump.py

conf-report:
	$(PYTHON) src/ingest/conf_abstract_report.py

conf-scan: dump conf-report

# analysis (analysis team)
topics:
	$(PYTHON) src/analysis/topics.py

yearly-topics:
	$(PYTHON) src/analysis/yearly_topics.py

clean-cache:
	rm -rf data outputs
