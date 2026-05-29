"""
ingest.py — Fetch DBLP metadata per year and cache as bronze JSON.

Writes one file per year to data/raw/<venue>_<year>.json with a bare list of
DBLP hits. clean.py reads these.

History: DBLP's `stream:streams/<slug>:` query returned 7,870 ICSE hits in
one paginated pass until 2026-05-29, when it started returning HTTP 500 for
unclear reasons. We now use the per-year `venue:<acronym> year:<Y>` query
instead, filtered to papers whose DBLP key starts with the slug (avoids
false-positive matches from other venues whose acronym shares a prefix).
"""
import json
import time
from pathlib import Path

import requests
import yaml

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DBLP_URL = "https://dblp.org/search/publ/api"
PAGE_SIZE = 1000
SLEEP_BETWEEN_CALLS = 3.0  # DBLP throttles aggressive clients aggressively
HEADERS = {
    "User-Agent": "TopicDrift-Research/0.1 (CS4530 project; mailto:maaseide.m@northeastern.edu)",
    "Accept": "application/json",
}


def load_venues():
    with open("config/venues.yaml") as f:
        return yaml.safe_load(f)


def _get_with_retry(params, max_attempts=7):
    for attempt in range(max_attempts):
        try:
            r = requests.get(DBLP_URL, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            wait = min(180, 2 ** attempt * 10)
            print(f"    DBLP connection error ({type(e).__name__}) — backing off {wait}s")
            time.sleep(wait)
            continue
        if r.status_code in (429, 500, 503):
            wait = min(180, 2 ** attempt * 10)
            print(f"    DBLP {r.status_code} — backing off {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"DBLP failed after {max_attempts} attempts (params={params})")


def fetch_year(slug, year):
    """All DBLP hits for (slug, year), paginated and filtered to the slug."""
    acronym = slug.rsplit("/", 1)[-1]
    prefix = slug.lower() + "/"
    hits = []
    offset = 0
    while True:
        data = _get_with_retry({
            "q": f"venue:{acronym} year:{year}",
            "format": "json",
            "h": PAGE_SIZE,
            "f": offset,
        })
        hits_obj = data.get("result", {}).get("hits", {})
        page = hits_obj.get("hit", [])
        total = int(hits_obj.get("@total", 0))
        hits.extend(page)
        offset += PAGE_SIZE
        if offset >= total or not page:
            break
        time.sleep(SLEEP_BETWEEN_CALLS)
    return [h for h in hits
            if (h.get("info", {}).get("key") or "").lower().startswith(prefix)]


def fetch_all_papers(venue_key):
    cfg = load_venues()[venue_key]
    slug = cfg["dblp_slugs"][0]
    start, end = cfg["start_year"], cfg["end_year"]

    total = 0
    written = 0
    for year in range(start, end + 1):
        path = RAW_DIR / f"{venue_key}_{year}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            print(f"  [cache] {path.name} — {len(cached)} papers")
            total += len(cached)
            written += 1
            continue
        hits = fetch_year(slug, year)
        if hits:
            path.write_text(json.dumps(hits, indent=2))
            written += 1
            total += len(hits)
        print(f"  [fetch] {venue_key} year={year} — {len(hits)} papers")
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nDone. {total} papers fetched for {venue_key}, {written} year files written")
    return total


if __name__ == "__main__":
    fetch_all_papers("icse")
