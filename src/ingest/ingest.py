import requests
import json
import time
from collections import defaultdict
from pathlib import Path
import yaml

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_venues():
    with open("config/venues.yaml") as f:
        return yaml.safe_load(f)


def fetch_all_papers(venue_key):
    venues = load_venues()
    config = venues[venue_key]
    slug = config["dblp_slugs"][0]

    all_hits = []
    batch_size = 500
    offset = 0

    while True:
        url = f"https://dblp.org/search/publ/api?q=stream:streams/{slug}:&format=json&h={batch_size}&f={offset}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        total = int(data.get("result", {}).get("hits", {}).get("@total", 0))

        print(f"  [fetch] {venue_key} offset={offset} — {len(hits)} papers (total={total})")

        all_hits.extend(hits)

        if offset + batch_size >= total:
            break

        offset += batch_size
        time.sleep(1)

    by_year = defaultdict(list)
    for hit in all_hits:
        year = hit.get("info", {}).get("year", "unknown")
        by_year[year].append(hit)

    for year, hits in sorted(by_year.items()):
        path = RAW_DIR / f"{venue_key}_{year}.json"
        path.write_text(json.dumps(hits, indent=2))
        print(f"  [write] {path.name} — {len(hits)} papers")

    print(f"\nDone. {len(all_hits)} papers fetched for {venue_key}, {len(by_year)} year files written")
    return all_hits


if __name__ == "__main__":
    fetch_all_papers("icse")