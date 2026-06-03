"""
enrich_acm.py — Recover abstracts for DOI-less A1 records via ACM DL HTML scrape.

Targets rows in data/interim/<venue>_enriched.parquet whose `ee` field points to
dl.acm.org / portal.acm.org but still have no abstract after the OpenAlex pass.

Auth — cookie-based (ACM has no public API for abstracts):
  1. Log in to https://dl.acm.org via your institution in a browser.
  2. Export cookies to a local file using a browser extension such as
     "Get cookies.txt LOCALLY" (saves Netscape/cookies.txt format) or
     "Cookie-Editor" (saves JSON format).
  3. Save the file to ~/.config/topic-drift/acm_cookies.json (or .txt),
     or set ACM_COOKIES_PATH=/path/to/your/cookies.<ext>.

Reads:  data/interim/<venue>_enriched.parquet   (from enrich_openalex.py)
Writes: data/interim/<venue>_enriched.parquet   (in-place, adds abstracts)

Cache: raw HTML → data/raw/acm/<sha1(canonical_url)>.html
       Confirmed misses → data/raw/acm/<sha1(canonical_url)>.miss
"""

import json
import os
import re
import sys
import time
from http.cookiejar import MozillaCookieJar
from pathlib import Path

import logging

import pandas as pd
import requests
from lxml import html as lxml_html

from topicdrift.ingest._matching import (
    _loose_title_match,
    _recompute_text_fields,
    _strict_title_match,
)
from topicdrift.utils.cache import make_cache_key

log = logging.getLogger(__name__)

INTERIM_DIR = Path("data/interim")
ACM_CACHE = Path("data/raw/acm")
ACM_CACHE.mkdir(parents=True, exist_ok=True)

DEFAULT_COOKIES = Path.home() / ".config" / "topic-drift" / "acm_cookies.json"
COOKIES_PATH = Path(os.environ.get("ACM_COOKIES_PATH", str(DEFAULT_COOKIES)))

RATE_LIMIT = 0.5  # req/s — polite to ACM
BACKOFF_CODES = {429, 503}
ACM_URL_PATTERN = re.compile(r"dl\.acm\.org|portal\.acm\.org", re.I)


def _cache_path(url: str) -> Path:
    return ACM_CACHE / f"{make_cache_key(url)}.html"


def _miss_path(url: str) -> Path:
    return ACM_CACHE / f"{make_cache_key(url)}.miss"


def load_session(cookies_path: Path) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    if not cookies_path.exists():
        raise SystemExit(
            f"\nCookie file not found: {cookies_path}\n\n"
            "To export ACM DL cookies:\n"
            "  1. Log into https://dl.acm.org/ via your institution in a browser.\n"
            "  2. Export cookies as json for dl.acm.org and save to:\n"
            f"     {cookies_path}\n"
            "  4. Set ACM_COOKIES_PATH env var if you use a different path.\n"
        )

    suffix = cookies_path.suffix.lower()
    if suffix == ".txt":
        jar = MozillaCookieJar(str(cookies_path))
        jar.load(ignore_discard=True, ignore_expires=True)
        session.cookies = jar  # type: ignore[assignment]
    else:
        raw = json.loads(cookies_path.read_text())
        if isinstance(raw, list):
            for c in raw:
                session.cookies.set(
                    c["name"],
                    c["value"],
                    domain=c.get("domain", ".dl.acm.org"),
                )
        elif isinstance(raw, dict):
            for name, value in raw.items():
                session.cookies.set(name, str(value))
        else:
            raise SystemExit(f"Unrecognized cookie file format: {cookies_path}")

    return session


def _resolve_canonical(session: requests.Session, url: str) -> str | None:
    """Follow redirects to get the canonical dl.acm.org/doi/... URL."""
    try:
        r = session.head(url, allow_redirects=True, timeout=15)
        return r.url
    except Exception as e:
        log.warning("    resolve failed: %s", e)
        return None


_last_request: list[float] = [0.0]


def _fetch_page(session: requests.Session, url: str) -> bytes | None:
    """Rate-limited GET with HTML caching and negative-miss sentinel."""
    elapsed = time.monotonic() - _last_request[0]
    gap = 1.0 / RATE_LIMIT
    if elapsed < gap:
        time.sleep(gap - elapsed)
    _last_request[0] = time.monotonic()

    cache = _cache_path(url)
    if cache.exists():
        return cache.read_bytes()
    if _miss_path(url).exists():
        return None

    for attempt in range(4):
        try:
            r = session.get(url, timeout=30, allow_redirects=True)
        except Exception as e:
            log.warning("    network error: %s", e)
            _miss_path(url).write_text("")
            return None

        if r.status_code == 200:
            cache.write_bytes(r.content)
            return r.content
        elif r.status_code == 403:
            log.error("  403 — session expired or not authenticated. Re-export cookies and retry.")
            sys.exit(1)
        elif r.status_code in BACKOFF_CODES:
            wait = 3 * 2**attempt
            log.warning(
                "    HTTP %d, backing off %ds (attempt %d/4)", r.status_code, wait, attempt + 1
            )
            time.sleep(wait)
        elif r.status_code == 404:
            _miss_path(url).write_text("")
            return None
        else:
            log.warning("    HTTP %d", r.status_code)
            _miss_path(url).write_text("")
            return None

    log.error("  Repeated rate-limit errors — stopping to protect session.")
    sys.exit(1)


def _extract_abstract(content: bytes) -> str | None:
    tree = lxml_html.fromstring(content)

    # ACM DL current layout: <section id="abstract"><div role="paragraph">...</div>
    paras = tree.xpath('//section[@id="abstract"]//*[@role="paragraph"]')
    if paras:
        text = " ".join(
            p.text_content().strip()
            for p in paras
            if p.text_content().strip().lower() != "abstract"
        ).strip()
        if text:
            return text

    # Older ACM DL layout: div.abstractSection
    for selector in (
        '//div[contains(@class,"abstractSection") and contains(@class,"abstractInFull")]//p',
        '//div[contains(@class,"abstractSection")]//p',
    ):
        divs = tree.xpath(selector)
        if divs:
            parts = [
                t
                for p in divs
                if (t := (p.text_content() or "").strip()) and t.lower() != "abstract"
            ]
            text = " ".join(parts).strip()
            if text:
                return text

    # Fallback: grab full section text and strip heading
    sec = tree.xpath('//section[@id="abstract"]')
    if sec:
        text = (sec[0].text_content() or "").strip()
        if text.lower().startswith("abstract"):
            text = text[len("abstract") :].strip()
        if text:
            return text

    return None


def _extract_title(content: bytes) -> str | None:
    tree = lxml_html.fromstring(content)

    # og:title / <title> both include " | Proceedings of..." suffix — strip it
    for selector in (
        '//meta[@property="og:title"]/@content',
        '//meta[@name="citation_title"]/@content',
    ):
        vals = tree.xpath(selector)
        if vals:
            t = vals[0].strip()
            if " | " in t:
                t = t.split(" | ")[0].strip()
            if t:
                return t

    title_els = tree.xpath("//title")
    if title_els:
        t = (title_els[0].text_content() or "").strip()
        if " | " in t:
            t = t.split(" | ")[0].strip()
        if t:
            return t

    return None


def _doi_from_url(url: str) -> str | None:
    if "/doi/" in url:
        part = url.split("/doi/", 1)[1].split("?")[0].strip("/")
        return part.lower() or None
    return None


def scrape_one(
    session: requests.Session,
    ee_url: str,
    dblp_title: str,
    year: int,
) -> dict | None:
    canonical = _resolve_canonical(session, ee_url)
    if not canonical:
        return None

    content = _fetch_page(session, canonical)
    if not content:
        return None

    page_title = _extract_title(content)
    if page_title:
        work = {"display_name": page_title, "publication_year": year}
        if _strict_title_match(dblp_title, year, work):
            pass
        elif _loose_title_match(dblp_title, year, work):
            log.info("  LOOSE MATCH: acm[%d] %r <- %r", year, dblp_title[:55], page_title[:55])
        else:
            log.warning(
                "  TITLE MISMATCH: acm[%d] %r <- %r", year, dblp_title[:55], page_title[:55]
            )

    abstract = _extract_abstract(content)
    if not abstract:
        return None

    return {"abstract": abstract, "doi": _doi_from_url(canonical)}


def enrich_acm(venue_key: str) -> None:
    src = INTERIM_DIR / f"{venue_key}_enriched.parquet"
    if not src.exists():
        raise SystemExit(f"Not found: {src}. Run enrich_openalex.py first.")

    df = pd.read_parquet(src)
    log.info("Loaded %d rows from %s", len(df), src)

    targets = df[~df["has_abstract"] & df["ee"].fillna("").str.contains(ACM_URL_PATTERN)]
    log.info(
        "ACM scrape targets: %d rows (years %d–%d)",
        len(targets),
        int(targets["year"].min()),
        int(targets["year"].max()),
    )

    if targets.empty:
        log.info("Nothing to do.")
        return

    session = load_session(COOKIES_PATH)
    log.info("Loaded cookies from %s", COOKIES_PATH)

    recovered = 0
    for i, (idx, row) in enumerate(targets.iterrows(), 1):
        log.info("  [%d/%d] [%d] %r", i, len(targets), int(row["year"]), row["title"][:60])
        result = scrape_one(session, row["ee"], row["title"], int(row["year"]))
        if result:
            df.at[idx, "abstract"] = result["abstract"]
            if result["doi"]:
                df.at[idx, "doi"] = result["doi"]
                df.at[idx, "has_doi"] = True
            recovered += 1
            log.info("    OK — %d chars", len(result["abstract"]))
        else:
            log.info("    no abstract")

    _recompute_text_fields(df)
    df.to_parquet(src, index=False)
    log.info("Recovered %d/%d abstracts", recovered, len(targets))
    log.info(
        "Wrote %s (%d rows, abstract coverage %.1f%%)",
        src,
        len(df),
        100 * df["has_abstract"].mean(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    venues = sys.argv[1:] or ["icse"]
    for v in venues:
        enrich_acm(v)
