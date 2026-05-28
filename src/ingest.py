import requests
import json
import time
from pathlib import Path
import yaml

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_venues():
    with open("config/venues.yaml") as f:
        return yaml.safe_load(f)


def fetch_venue_year(slug, year):
    cache_path = RAW_DIR / f"{slug.replace('/', '_')}_{year}.json"

    if cache_path.exists():
        print(f"  [cache] {slug} {year}")
        return json.loads(cache_path.read_text())

    url = "https://dblp.org/search/publ/api"
    params = {
        "q": f"{slug} {year}",
        "format": "json",
        "h": 500
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    cache_path.write_text(json.dumps(data, indent=2))

    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    print(f"  [fetch] {slug} {year} — {len(hits)} papers")

    time.sleep(1)
    return data


def ingest_venue(venue_key):
    venues = load_venues()
    config = venues[venue_key]

    for slug in config["dblp_slugs"]:
        print(f"\nFetching {slug} ({config['start_year']}–{config['end_year']})")
        for year in range(config["start_year"], config["end_year"] + 1):
            fetch_venue_year(slug, year)


if __name__ == "__main__":
    ingest_venue("icse")
