"""
enrich_openalex.py — Enrich the interim DBLP parquet with OpenAlex data.

Two invocation modes
--------------------
Default (DOI pass only — fast, runs on any corpus size):
    make venue  (or: python -m topicdrift.ingest.enrich_openalex [venue ...])

    Reads:  data/interim/<venue>_dblp.parquet      (from venue.py via `make venue`)
    Writes: data/interim/<venue>_enriched.parquet

    Fetches DOI-matched works from OpenAlex in parallel batches.
    Also applies any manual overrides from data/manual/<venue>_overrides.csv.
    When no venues are given, processes ALL *_dblp.parquet files found in
    data/interim/.

Title pass (slow — one API call per DOI-less paper; not feasible at >100K papers):
    make titles  (or: python -m topicdrift.ingest.enrich_openalex --title-pass [venue ...])

    Reads:  data/interim/<venue>_enriched.parquet  (must exist from DOI pass)
    Writes: data/interim/<venue>_enriched.parquet  (in-place, adds abstracts)

    Runs OpenAlex title.search for each row still missing an abstract after the
    DOI pass. Intended for targeted use on small corpora (single venue).

Columns added by the DOI pass:
    venue           str           venue key (e.g. "icse")
    abstract        str | None    reconstructed from OpenAlex inverted index
    has_abstract    bool          recomputed from `abstract`
    text            str           normalized title + abstract (TF-IDF ready)
    oa_concepts     list[str]     concept tags with score >= 0.3
    citation_count  int | None    cited_by_count
    openalex_id     str | None
    oa_type         str | None    e.g. "article"

OpenAlex polite pool (mailto=): 10 req/s. A doi filter takes at most 100 values
and the GET URL must stay under 4094 bytes, so batches are packed to both limits.
Raw responses cached under data/raw/openalex/.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from topicdrift.ingest._matching import (
    _loose_title_match,
    _recompute_text_fields,
    _strict_title_match,
)
from topicdrift.utils.cache import make_cache_key
from topicdrift.utils.doi import normalize_doi
from topicdrift.utils.http import (
    MAILTO,
    MAX_WORKERS,
    OPENALEX_URL,
    SELECT_FIELDS,
    RateLimiter,
    fetch_openalex_batch,
    get_session,
    pack_batches,
)

log = logging.getLogger(__name__)

INTERIM_DIR = Path("data/interim")
OA_CACHE = Path("data/raw/openalex")
OA_CACHE.mkdir(parents=True, exist_ok=True)

MAX_REQUESTS_PER_SEC = 10
CONCEPT_MIN_SCORE = 0.3

_rate_limiter = RateLimiter(MAX_REQUESTS_PER_SEC)


def reconstruct_abstract(inv_index: dict | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]} — rebuild the text."""
    if not inv_index:
        return None
    positions = sorted((i, w) for w, idxs in inv_index.items() for i in idxs)
    return " ".join(w for _, w in positions) or None


def _cache_path(dois: list[str]) -> Path:
    return OA_CACHE / f"{make_cache_key(dois)}.json"


def _title_cache_path(title: str, year: int) -> Path:
    return OA_CACHE / f"title_{make_cache_key(f'{title}|{year}')}.json"


def _oa_title_params(title: str, year: int) -> dict:
    safe_title = title.replace('"', '\\"')
    return {
        "filter": f'title.search:"{safe_title}",publication_year:{year}',
        "per-page": 5,
        "mailto": MAILTO,
        "select": SELECT_FIELDS,
    }


def _fetch_batch(dois: list[str]) -> list[dict]:
    result = fetch_openalex_batch(dois, SELECT_FIELDS, _cache_path, _rate_limiter)
    if result is None:
        raise RuntimeError(f"Batch of {len(dois)} DOIs failed after repeated 429s")
    return result


def parse_work(work: dict) -> dict:
    return {
        "doi": normalize_doi(work.get("doi")),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "oa_concepts": [
            c["display_name"]
            for c in (work.get("concepts") or [])
            if (c.get("score") or 0) >= CONCEPT_MIN_SCORE
        ],
        "citation_count": work.get("cited_by_count"),
        "openalex_id": work.get("id"),
        "oa_type": work.get("type"),
    }


def _fetch_title_one(idx: int, title: str, year: int) -> tuple[int, dict | None, str | None]:
    """OpenAlex title.search for one row. Prefers a strict normalized-title + year
    match; falls back to a loose match (logged for audit). Returns the match kind."""
    cache = _title_cache_path(title, year)
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        data = {"results": []}
        for attempt in range(4):
            _rate_limiter.acquire()
            r = get_session().get(OPENALEX_URL, params=_oa_title_params(title, year), timeout=30)
            if r.status_code == 429:
                time.sleep(3 * 2**attempt)
                continue
            r.raise_for_status()
            data = r.json()
            cache.write_text(json.dumps(data, ensure_ascii=False))
            break
        else:
            raise RuntimeError(f"Title lookup [{year}] {title[:60]!r} failed after repeated 429s")

    results = data.get("results", [])
    for work in results:
        if _strict_title_match(title, year, work):
            return idx, parse_work(work), "strict"
    for work in results:
        if _loose_title_match(title, year, work):
            log.info(
                "LOOSE MATCH: dblp[%d] %r <- oa[%s] %r (%s)",
                year,
                title[:55],
                work.get("publication_year"),
                (work.get("display_name") or "")[:55],
                work.get("id"),
            )
            return idx, parse_work(work), "loose"
    return idx, None, None


def fetch_by_titles(rows: list[tuple[int, str, int]]) -> dict[int, dict]:
    """Parallel OpenAlex title pass for DOI-less rows. Keys are dataframe indices."""
    if not rows:
        return {}

    log.info("  %d rows in title pass, %d workers", len(rows), MAX_WORKERS)
    t0 = time.monotonic()
    out: dict[int, dict] = {}
    strict_n = loose_n = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(_fetch_title_one, idx, title, year) for idx, title, year in rows]
        for fut in as_completed(futs):
            idx, parsed, kind = fut.result()
            if parsed:
                out[idx] = parsed
                strict_n += kind == "strict"
                loose_n += kind == "loose"

    log.info(
        "  matched %d/%d titles (%d strict, %d loose) in %.1fs",
        len(out),
        len(rows),
        strict_n,
        loose_n,
        time.monotonic() - t0,
    )
    return out


def _empty_enrichment() -> dict:
    return {
        "abstract": None,
        "citation_count": None,
        "openalex_id": None,
        "oa_type": None,
        "oa_concepts": [],
    }


def _apply_enrichment(out: pd.DataFrame, idx: int, data: dict) -> None:
    for col in ("abstract", "citation_count", "openalex_id", "oa_type", "oa_concepts"):
        out.at[idx, col] = data.get(col)
    if data.get("doi"):
        out.at[idx, "doi"] = data["doi"]


def _index_cached_works() -> dict[str, dict]:
    """Index every cached OpenAlex response by DOI. Batch cache files are keyed
    by their exact DOI set, so changing the input (e.g. dropping rows) re-batches
    and would otherwise refetch everything; this recovers those results offline."""
    out: dict[str, dict] = {}
    for path in OA_CACHE.glob("*.json"):
        if path.name.startswith("title_"):
            continue
        try:
            results = json.loads(path.read_text()).get("results", [])
        except (json.JSONDecodeError, OSError):
            continue
        for work in results:
            parsed = parse_work(work)
            if parsed["doi"]:
                out[parsed["doi"]] = parsed
    return out


def fetch_for_dois(dois: list[str]) -> dict[str, dict]:
    uniq = sorted({d for d in dois if d})
    cached = _index_cached_works()
    out = {d: cached[d] for d in uniq if d in cached}
    remaining = [d for d in uniq if d not in out]

    if remaining:
        batches = pack_batches(remaining)
        log.info(
            "  %d/%d DOIs uncached, %d batches, %d workers",
            len(remaining),
            len(uniq),
            len(batches),
            MAX_WORKERS,
        )
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for fut in as_completed(pool.submit(_fetch_batch, b) for b in batches):
                for work in fut.result():
                    parsed = parse_work(work)
                    if parsed["doi"]:
                        out[parsed["doi"]] = parsed
        log.info("  fetched remainder in %.1fs", time.monotonic() - t0)

    log.info("  matched %d/%d DOIs", len(out), len(uniq))
    return out


def enrich_venue(venue_key: str) -> None:
    """DOI pass: fetch OpenAlex data for all DOI-bearing rows, apply overrides."""
    src = INTERIM_DIR / f"{venue_key}_dblp.parquet"
    if not src.exists():
        raise SystemExit(f"Not found: {src}. Run `make venue VENUE={venue_key}` first.")

    df = pd.read_parquet(src)
    log.info("[%s] Loaded %d rows from %s", venue_key, len(df), src)

    enriched = fetch_for_dois(df["doi"].dropna().tolist())
    empty = _empty_enrichment()
    add = pd.DataFrame(enriched.get(d, empty) for d in df["doi"])
    add = add.drop(columns="doi", errors="ignore")

    out = pd.concat([df.reset_index(drop=True), add.reset_index(drop=True)], axis=1)
    out["venue"] = venue_key
    _recompute_text_fields(out)

    n_doi_less = int((out["doi"].isna() & out["abstract"].isna()).sum())
    if n_doi_less:
        log.info(
            "[%s] %d DOI-less rows without abstract (run --title-pass to attempt recovery)",
            venue_key,
            n_doi_less,
        )

    overrides_path = Path("data/manual") / f"{venue_key}_overrides.csv"
    if overrides_path.exists():
        overrides = pd.read_csv(overrides_path).dropna(subset=["abstract"])
        if not overrides.empty:
            key_to_idx = out.reset_index().set_index("dblp_key")["index"]
            applied = 0
            for _, ov in overrides.iterrows():
                if ov["dblp_key"] in key_to_idx.index:
                    idx = key_to_idx[ov["dblp_key"]]
                    out.at[idx, "abstract"] = ov["abstract"]
                    applied += 1
            if applied:
                _recompute_text_fields(out)
                log.info(
                    "[%s] Applied %d manual override(s) from %s",
                    venue_key,
                    applied,
                    overrides_path,
                )

    dest = INTERIM_DIR / f"{venue_key}_enriched.parquet"
    out.to_parquet(dest, index=False)
    log.info(
        "[%s] Wrote %s (%d rows, abstract coverage %.1f%%)",
        venue_key,
        dest,
        len(out),
        100 * out["has_abstract"].mean(),
    )


def enrich_venue_titles(venue_key: str) -> None:
    """Title pass: OpenAlex title.search for DOI-less rows still missing an abstract.

    Reads and overwrites data/interim/<venue>_enriched.parquet in place.
    One API call per row — only use this on small corpora.
    """
    src = INTERIM_DIR / f"{venue_key}_enriched.parquet"
    if not src.exists():
        raise SystemExit(f"Not found: {src}. Run the DOI pass first.")

    out = pd.read_parquet(src)
    log.info("[%s] Loaded %d rows from %s", venue_key, len(out), src)

    doi_less = out["doi"].isna() & out["abstract"].isna()
    n_doi_less = int(doi_less.sum())

    if not n_doi_less:
        log.info("[%s] No DOI-less rows without abstract — nothing to do.", venue_key)
        return

    log.info("[%s] %d DOI-less rows without abstract — OpenAlex title pass", venue_key, n_doi_less)
    fallback_rows = [
        (idx, out.at[idx, "title"], int(out.at[idx, "year"])) for idx in out.index[doi_less]
    ]
    recovered = fetch_by_titles(fallback_rows)
    for idx, data in recovered.items():
        _apply_enrichment(out, idx, data)
    _recompute_text_fields(out)

    out.to_parquet(src, index=False)
    log.info(
        "[%s] Recovered %d/%d abstracts via title search (%.1f%%)",
        venue_key,
        len(recovered),
        n_doi_less,
        100 * len(recovered) / n_doi_less,
    )
    log.info("[%s] Abstract coverage now %.1f%%", venue_key, 100 * out["has_abstract"].mean())


def _all_venue_keys() -> list[str]:
    """Return all venue keys with a *_dblp.parquet file in INTERIM_DIR."""
    return sorted(p.stem.removesuffix("_dblp") for p in INTERIM_DIR.glob("*_dblp.parquet"))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = sys.argv[1:]
    title_pass = "--title-pass" in args
    venues = [a for a in args if not a.startswith("--")]

    if not venues:
        if title_pass:
            venues = sorted(
                p.stem.removesuffix("_enriched") for p in INTERIM_DIR.glob("*_enriched.parquet")
            )
        else:
            venues = _all_venue_keys()

    if not venues:
        raise SystemExit("No venues found. Run `make venue` first.")

    if title_pass:
        for v in venues:
            enrich_venue_titles(v)
    else:
        for v in venues:
            enrich_venue(v)
