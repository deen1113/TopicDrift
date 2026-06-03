"""Shared HTTP utilities: thread-local sessions, rate limiting, and OpenAlex batch fetch."""

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode

import requests

_thread_local = threading.local()

# OpenAlex polite-pool endpoint; mailto= unlocks 10 req/s.
OPENALEX_URL = "https://api.openalex.org/works"
MAILTO = "maaseide.m@northeastern.edu"
MAX_DOIS = 100  # OpenAlex doi filter value cap
MAX_URL_LEN = 4080  # OpenAlex rejects above 4094 bytes
MAX_WORKERS = 16  # default parallel threads for batch fetching
SELECT_FIELDS = (  # full field set used by the enrichment pass
    "id,doi,display_name,publication_year,abstract_inverted_index,concepts,cited_by_count,type"
)


class RateLimiter:
    """Spaces calls across threads to honor a req/s cap."""

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


def get_session() -> requests.Session:
    """Return a per-thread requests.Session (created lazily)."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def _is_budget_error(r: requests.Response) -> bool:
    return r.status_code == 429 and "budget" in r.text.lower()


def _backoff_429(r: requests.Response, attempt: int) -> None:
    """Honor Retry-After if present, otherwise exponential backoff."""
    retry_after = r.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        time.sleep(min(120, int(retry_after)))
    else:
        time.sleep(min(60, 2**attempt))


def _build_params(dois: list[str], select: str) -> dict:
    return {
        "filter": "doi:" + "|".join(dois),
        "per-page": len(dois),
        "mailto": MAILTO,
        "select": select,
    }


def _url_len(dois: list[str]) -> int:
    return len(OPENALEX_URL) + 1 + len(urlencode(_build_params(dois, SELECT_FIELDS)))


def pack_batches(dois: list[str]) -> list[list[str]]:
    """Pack DOIs into batches: ≤ MAX_DOIS values and URL ≤ MAX_URL_LEN per call."""
    batches, batch = [], []
    for doi in dois:
        if batch and (len(batch) == MAX_DOIS or _url_len(batch + [doi]) > MAX_URL_LEN):
            batches.append(batch)
            batch = []
        batch.append(doi)
    if batch:
        batches.append(batch)
    return batches


def fetch_openalex_batch(
    dois: list[str],
    select: str,
    cache_fn: Callable[[list[str]], Path],
    rate_limiter: RateLimiter,
    *,
    max_attempts: int = 4,
    budget_event: threading.Event | None = None,
) -> list[dict] | None:
    """Fetch one OpenAlex DOI-filter batch, caching the raw response.

    Returns the results list on success (may be []), or None on budget
    exhaustion or exhausted retries. Bisects on HTTP 400 automatically.
    The caller decides whether None is a hard error or a soft skip.
    """
    if not dois:
        return []
    cache = cache_fn(dois)
    if cache.exists():
        return json.loads(cache.read_text()).get("results", [])
    if budget_event is not None and budget_event.is_set():
        return None

    params = _build_params(dois, select)
    for attempt in range(max_attempts):
        rate_limiter.acquire()
        try:
            r = get_session().get(OPENALEX_URL, params=params, timeout=(10, 60))
        except requests.RequestException:
            time.sleep(min(60, 2**attempt))
            continue
        if r.status_code == 429:
            if budget_event is not None and _is_budget_error(r):
                budget_event.set()
                return None
            _backoff_429(r, attempt)
            continue
        if r.status_code == 400 and len(dois) > 1:
            mid = len(dois) // 2
            left = fetch_openalex_batch(
                dois[:mid],
                select,
                cache_fn,
                rate_limiter,
                max_attempts=max_attempts,
                budget_event=budget_event,
            )
            right = fetch_openalex_batch(
                dois[mid:],
                select,
                cache_fn,
                rate_limiter,
                max_attempts=max_attempts,
                budget_event=budget_event,
            )
            if left is None or right is None:
                return None
            return left + right
        if r.status_code >= 500:
            time.sleep(min(60, 2**attempt))
            continue
        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(data, ensure_ascii=False))
        return data.get("results", [])

    return None
