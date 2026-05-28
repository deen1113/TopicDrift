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


def fetch_all_papers(venue_key):
    venues = load_venues()
    config = venues[venue_key]
    slug = config["dblp_slugs"][0]

    all_hits = []
    batch_size = 500
    offset = 0

    while True:
        cache_path = RAW_DIR / f"{venue_key}_dblp_f{offset}.json"

        if cache_path.exists():
            print(f"  [cache] {venue_key} offset={offset}")
            data = json.loads(cache_path.read_text())
        else:
            params = {
                "q": f"stream:streams/{slug}:",
                "format": "json",
                "h": batch_size,
                "f": offset
            }
            response = requests.get("https://dblp.org/search/publ/api", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            cache_path.write_text(json.dumps(data, indent=2))
            time.sleep(1)

        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        total = int(data.get("result", {}).get("hits", {}).get("@total", 0))

        print(f"  [fetch] {venue_key} offset={offset} — {len(hits)} papers (total={total})")

        all_hits.extend(hits)

        if offset + batch_size >= total:
            break

        offset += batch_size

    print(f"\nDone. {len(all_hits)} papers fetched for {venue_key}")
    return all_hits


if __name__ == "__main__":
    fetch_all_papers("icse")