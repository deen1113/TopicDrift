"""
enrich_openalex.py — Enrich the interim DBLP parquet with OpenAlex data.

Reads:  data/interim/<venue>_dblp.parquet      (from src/clean.py)
Writes: data/interim/<venue>_enriched.parquet

Adds columns:
    venue           str           venue key (e.g. "icse")
    abstract        str | None    reconstructed from OpenAlex inverted index
    has_abstract    bool          recomputed from `abstract`
    text            str           normalized title + abstract (TF-IDF ready)
    oa_concepts     list[str]     concept tags with score >= 0.3
    citation_count  int | None    cited_by_count
    openalex_id     str | None
    oa_type         str | None    e.g. "article"

OpenAlex polite pool (mailto=): 10 req/s. A doi filter takes at most 100 values
and the GET URL must stay under 4094 bytes, so batches are packed to both.
DOI-less rows get a second pass via OpenAlex title.search + publication_year.
Raw responses cached under data/raw/openalex/.
"""
import hashlib
import json
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

INTERIM_DIR = Path("data/interim")
OA_CACHE = Path("data/raw/openalex")
OA_CACHE.mkdir(parents=True, exist_ok=True)

OPENALEX_URL = "https://api.openalex.org/works"
MAILTO = "maaseide.m@northeastern.edu"
MAX_URL_LEN = 4080  # OpenAlex rejects above 4094 bytes
MAX_DOIS = 100  # OpenAlex doi filter value cap
MAX_REQUESTS_PER_SEC = 10
MAX_WORKERS = 16
CONCEPT_MIN_SCORE = 0.3
SELECT_FIELDS = (
    "id,doi,display_name,publication_year,abstract_inverted_index,"
    "concepts,cited_by_count,type"
)

LATEX = re.compile(r"\$[^$]*\$")
HTML = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")

# DBLP often prepends "On"/"On a"/"On the" to titles that OpenAlex stores
# without. Strip these (and bare articles) so the loose matcher can equate them.
LEADING_ARTICLES = re.compile(r"^(?:on\s+(?:a\s+|an\s+|the\s+)?|a\s+|an\s+|the\s+)+")

_thread_local = threading.local()


class RateLimiter:
    """Spaces calls across all threads to honor the polite-pool req/s cap."""

    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now += wait
            self._next = now + self._interval


_rate_limiter = RateLimiter(MAX_REQUESTS_PER_SEC)


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def normalize(text: str | None) -> str:
    """Lowercase, drop LaTeX/HTML, collapse whitespace; "" for non-strings."""
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = HTML.sub(" ", LATEX.sub(" ", s))
    return WS.sub(" ", s.lower()).strip()


def reconstruct_abstract(inv_index: dict | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]} — rebuild the text."""
    if not inv_index:
        return None
    positions = sorted((i, w) for w, idxs in inv_index.items() for i in idxs)
    return " ".join(w for _, w in positions) or None


def _params(dois: list[str]) -> dict:
    return {
        "filter": "doi:" + "|".join(dois),
        "per-page": len(dois),
        "mailto": MAILTO,
        "select": SELECT_FIELDS,
    }


def _url_len(dois: list[str]) -> int:
    return len(OPENALEX_URL) + 1 + len(urlencode(_params(dois)))


def _pack_batches(dois: list[str]) -> list[list[str]]:
    """Greedy pack: ≤ MAX_DOIS values and encoded URL ≤ MAX_URL_LEN per call."""
    batches, batch = [], []
    for doi in dois:
        if batch and (len(batch) == MAX_DOIS or _url_len(batch + [doi]) > MAX_URL_LEN):
            batches.append(batch)
            batch = []
        batch.append(doi)
    if batch:
        batches.append(batch)
    return batches


def _cache_path(dois: list[str]) -> Path:
    digest = hashlib.sha1("|".join(sorted(dois)).encode()).hexdigest()
    return OA_CACHE / f"{digest}.json"


def _title_cache_path(title: str, year: int) -> Path:
    digest = hashlib.sha1(f"{title}|{year}".encode()).hexdigest()
    return OA_CACHE / f"title_{digest}.json"


def _oa_title_params(title: str, year: int) -> dict:
    safe_title = title.replace('"', '\\"')
    return {
        "filter": f'title.search:"{safe_title}",publication_year:{year}',
        "per-page": 5,
        "mailto": MAILTO,
        "select": SELECT_FIELDS,
    }


def _strict_title_match(title: str, year: int, work: dict) -> bool:
    if work.get("publication_year") != year:
        return False
    work_title = work.get("display_name") or ""
    return normalize(work_title) == normalize(title)


def _loose_key(text: str | None) -> str:
    """normalize(), strip punctuation, then strip leading articles (DBLP-style
    "On"/"On the" prefixes) so trivially-different titles compare equal."""
    s = WS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", normalize(text))).strip()
    return LEADING_ARTICLES.sub("", s)


def _loose_title_match(title: str, year: int, work: dict) -> bool:
    """Punctuation-insensitive title equality within +/-1 year (recovers year
    drift and trailing-punctuation differences that strict matching rejects)."""
    work_year = work.get("publication_year")
    if work_year is None or abs(work_year - year) > 1:
        return False
    return _loose_key(work.get("display_name")) == _loose_key(title)


def _fetch_batch(dois: list[str]) -> list[dict]:
    """Fetch one batch, caching raw results. Bisects on an unexpected 400."""
    if not dois:
        return []

    cache = _cache_path(dois)
    if cache.exists():
        return json.loads(cache.read_text()).get("results", [])

    for attempt in range(4):
        _rate_limiter.acquire()
        r = _session().get(OPENALEX_URL, params=_params(dois), timeout=30)
        if r.status_code == 429:
            time.sleep(3 * 2**attempt)
            continue
        if r.status_code == 400 and len(dois) > 1:
            mid = len(dois) // 2
            return _fetch_batch(dois[:mid]) + _fetch_batch(dois[mid:])
        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(data, ensure_ascii=False))
        return data.get("results", [])

    print(f"  giving up on batch of {len(dois)} after repeated 429s")
    return []


def parse_work(work: dict) -> dict:
    return {
        "doi": (work.get("doi") or "").replace("https://doi.org/", "").lower() or None,
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
            r = _session().get(
                OPENALEX_URL, params=_oa_title_params(title, year), timeout=30
            )
            if r.status_code == 429:
                time.sleep(3 * 2**attempt)
                continue
            r.raise_for_status()
            data = r.json()
            cache.write_text(json.dumps(data, ensure_ascii=False))
            break
        else:
            print(f"  giving up on title lookup [{year}] {title[:60]!r}")
            return idx, None, None

    results = data.get("results", [])
    for work in results:
        if _strict_title_match(title, year, work):
            return idx, parse_work(work), "strict"
    for work in results:
        if _loose_title_match(title, year, work):
            print(
                f"  LOOSE MATCH: dblp[{year}] {title[:55]!r} <- "
                f"oa[{work.get('publication_year')}] "
                f"{(work.get('display_name') or '')[:55]!r} ({work.get('id')})"
            )
            return idx, parse_work(work), "loose"
    return idx, None, None


def fetch_by_titles(rows: list[tuple[int, str, int]]) -> dict[int, dict]:
    """Parallel OpenAlex title pass for DOI-less rows. Keys are dataframe indices."""
    if not rows:
        return {}

    print(f"  {len(rows)} rows in title pass, {MAX_WORKERS} workers")
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

    print(
        f"  matched {len(out)}/{len(rows)} titles "
        f"({strict_n} strict, {loose_n} loose) in {time.monotonic() - t0:.1f}s"
    )
    return out


def _empty_enrichment() -> dict:
    empty = {f: None for f in ("abstract", "citation_count", "openalex_id", "oa_type")}
    empty["oa_concepts"] = []
    return empty


def _apply_enrichment(out: pd.DataFrame, idx: int, data: dict) -> None:
    for col in ("abstract", "citation_count", "openalex_id", "oa_type", "oa_concepts"):
        out.at[idx, col] = data.get(col)
    if data.get("doi"):
        out.at[idx, "doi"] = data["doi"]


def _recompute_text_fields(out: pd.DataFrame) -> None:
    out["has_abstract"] = out["abstract"].fillna("").str.len() > 0
    out["text"] = (
        out["title"].map(normalize) + " " + out["abstract"].map(normalize)
    ).str.strip()


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
        batches = _pack_batches(remaining)
        print(f"  {len(remaining)}/{len(uniq)} DOIs uncached, {len(batches)} batches, {MAX_WORKERS} workers")
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for fut in as_completed(pool.submit(_fetch_batch, b) for b in batches):
                for work in fut.result():
                    parsed = parse_work(work)
                    if parsed["doi"]:
                        out[parsed["doi"]] = parsed
        print(f"  fetched remainder in {time.monotonic() - t0:.1f}s")

    print(f"  matched {len(out)}/{len(uniq)} DOIs")
    return out


def enrich_venue(venue_key: str) -> None:
    src = INTERIM_DIR / f"{venue_key}_dblp.parquet"
    if not src.exists():
        raise SystemExit(f"Not found: {src}. Run `python src/clean.py` first.")

    df = pd.read_parquet(src)
    print(f"Loaded {len(df)} rows from {src}")

    enriched = fetch_for_dois(df["doi"].dropna().tolist())
    empty = _empty_enrichment()
    add = pd.DataFrame(enriched.get(d, empty) for d in df["doi"])
    add = add.drop(columns="doi", errors="ignore")

    out = pd.concat([df.reset_index(drop=True), add.reset_index(drop=True)], axis=1)
    out["venue"] = venue_key
    _recompute_text_fields(out)

    doi_less = out["doi"].isna() & out["abstract"].isna()
    n_doi_less = int(doi_less.sum())

    if n_doi_less:
        print(f"{n_doi_less} DOI-less rows without abstract — OpenAlex title pass")
        fallback_rows = [
            (idx, out.at[idx, "title"], int(out.at[idx, "year"]))
            for idx in out.index[doi_less]
        ]
        recovered = fetch_by_titles(fallback_rows)
        for idx, data in recovered.items():
            _apply_enrichment(out, idx, data)
        _recompute_text_fields(out)

    dest = INTERIM_DIR / f"{venue_key}_enriched.parquet"
    out.to_parquet(dest, index=False)
    print(f"Wrote {dest} ({len(out)} rows, abstract coverage {100 * out['has_abstract'].mean():.1f}%)")
    if n_doi_less:
        print(
            f"  DOI-less abstract rate: {100 * len(recovered) / n_doi_less:.1f}% "
            f"({len(recovered)}/{n_doi_less})"
        )


if __name__ == "__main__":
    enrich_venue("icse")
