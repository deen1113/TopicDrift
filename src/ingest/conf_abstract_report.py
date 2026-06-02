"""
conf_abstract_report.py — Rank every DBLP conference by abstract hit rate.

Reads:  data/interim/dblp_conf.parquet      (from ingest_dblp_dump.py)
Writes: outputs/tables/conf_abstract_hit_rate.csv      (venues >= 20 papers, ranked)
        outputs/tables/conf_abstract_hit_rate_all.csv  (every venue)
        outputs/reports/conf_abstract_hit_rate.md
        data/interim/dblp_doi_less.parquet             (all DOI-less papers)
        outputs/tables/dblp_doi_less_sample.csv        (readable sample)

"Abstract hit rate" = papers we can get an abstract for / total papers in the
venue. We use OpenAlex's DOI filter only (per-paper title.search doesn't scale to
~3M papers), so a paper counts as a hit iff it has a DOI *and* OpenAlex carries
an abstract for it. DOI-less papers count as misses — they're surfaced
separately so they can be recovered later.

We reuse the rate-limiter / batching / retry machinery from enrich_openalex but
request a slimmed payload (just enough to know if an abstract exists) and cache
to a separate dir so we don't disturb the richer ICSE enrichment cache.
"""
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from enrich_openalex import (
    MAILTO,
    MAX_WORKERS,
    OPENALEX_URL,
    RateLimiter,
    _pack_batches,
    _session,
)

INTERIM_DIR = Path("data/interim")
TABLES_DIR = Path("outputs/tables")
REPORTS_DIR = Path("outputs/reports")
OA_SCAN_CACHE = Path("data/raw/openalex_scan")
SCAN_SUMMARY = OA_SCAN_CACHE / "_scan_summary.json"  # merged doi->has_abstract map
for d in (TABLES_DIR, REPORTS_DIR, OA_SCAN_CACHE):
    d.mkdir(parents=True, exist_ok=True)

SRC = INTERIM_DIR / "dblp_conf.parquet"
MIN_VENUE_SIZE = 20  # floor for the ranked report
SAMPLE_ROWS = 5_000
# Only what we need to know whether OpenAlex has an abstract for a DOI.
SCAN_SELECT = "id,doi,abstract_inverted_index"

# Sustained whole-corpus scanning gets throttled at the polite-pool's nominal
# 10 req/s, so we run a touch under it with a dedicated limiter (the shared
# enrich_openalex limiter stays at 10/s for the smaller ICSE job). Failed
# batches aren't cached, so scan_dois retries them in later rounds until the
# whole corpus is cached — the report is only assembled from cache.
SCAN_RATE = 8.0
MAX_ATTEMPTS = 6   # per batch, per round
MAX_ROUNDS = 10    # whole-corpus retry passes for stragglers
_scan_limiter = RateLimiter(SCAN_RATE)

# OpenAlex's free tier enforces a daily spend budget (~$1, resets midnight UTC).
# When it runs out, requests 429 with an "Insufficient budget" message. There's
# no point retrying until the reset, so the first such response trips this flag
# and the whole run winds down fast; cached batches persist and a re-run the next
# day picks up where it left off. The report is only built once every batch is
# cached, so a budget-limited partial run never produces wrong hit rates.
_budget_exhausted = threading.Event()


def _is_budget_error(r: requests.Response) -> bool:
    return r.status_code == 429 and "budget" in r.text.lower()


def _params_slim(dois: list[str]) -> dict:
    return {
        "filter": "doi:" + "|".join(dois),
        "per-page": len(dois),
        "mailto": MAILTO,
        "select": SCAN_SELECT,
    }


def _scan_cache_path(dois: list[str]) -> Path:
    digest = hashlib.sha1("|".join(sorted(dois)).encode()).hexdigest()
    return OA_SCAN_CACHE / f"{digest}.json"


def _sleep_429(r: requests.Response, attempt: int) -> None:
    """Back off on a 429: honor Retry-After if present, else exponential."""
    retry_after = r.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        time.sleep(min(120, int(retry_after)))
    else:
        time.sleep(min(60, 2**attempt))


def _fetch_batch_slim(dois: list[str]) -> bool:
    """Fetch one slim DOI batch and cache it. Returns True if the result is
    cached (now or already), False if this round's attempts were exhausted (the
    batch stays uncached and a later round retries it). Bisects on a 400."""
    if not dois:
        return True
    cache = _scan_cache_path(dois)
    if cache.exists():
        return True
    if _budget_exhausted.is_set():
        return False

    for attempt in range(MAX_ATTEMPTS):
        _scan_limiter.acquire()
        try:
            r = _session().get(OPENALEX_URL, params=_params_slim(dois), timeout=(10, 60))
        except requests.RequestException:
            time.sleep(min(60, 2**attempt))
            continue
        if r.status_code == 429:
            if _is_budget_error(r):
                _budget_exhausted.set()
                return False
            _sleep_429(r, attempt)
            continue
        if r.status_code == 400 and len(dois) > 1:
            mid = len(dois) // 2
            return _fetch_batch_slim(dois[:mid]) and _fetch_batch_slim(dois[mid:])
        if r.status_code >= 500:
            time.sleep(min(60, 2**attempt))
            continue
        r.raise_for_status()
        cache.write_text(json.dumps(r.json(), ensure_ascii=False))
        return True

    return False


def _run_round(pending: list[list[str]]) -> int:
    """Fetch every pending batch in parallel; return how many failed."""
    t0 = time.monotonic()
    failed = done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for fut in as_completed(pool.submit(_fetch_batch_slim, b) for b in pending):
            failed += not fut.result()
            done += 1
            if done % 500 == 0:
                rate = done / max(time.monotonic() - t0, 1e-6)
                print(f"    {done:,}/{len(pending):,} ({rate:.1f}/s, {failed:,} failed)")
    return failed


def scan_dois(dois: list[str]) -> tuple[dict[str, bool], int, int]:
    """Map each DOI OpenAlex knows about -> whether it carries an abstract
    (DOIs absent from the result have no OpenAlex record at all).

    Caching and result assembly are decoupled: we run rounds until every batch
    is cached (or budget runs out / MAX_ROUNDS is hit), then read results from
    cache. A throttled or budget-limited round leaves its batches uncached for a
    later round/day, so a partial run never corrupts the counts.

    Returns (doi->has_abstract, n_cached_batches, n_total_batches)."""
    uniq = sorted({d for d in dois if d})
    batches = _pack_batches(uniq)
    n_total = len(batches)
    print(f"  {len(uniq):,} unique DOIs, {n_total:,} batches, "
          f"{MAX_WORKERS} workers @ {SCAN_RATE:.0f} req/s")

    # Fast path: if the merged summary exists and covers this exact DOI set,
    # skip re-reading all ~33k individual batch files.
    if SCAN_SUMMARY.exists():
        summary = json.loads(SCAN_SUMMARY.read_text())
        if summary.get("n_total") == n_total and summary.get("n_dois") == len(uniq):
            out: dict[str, bool] = summary["scan"]
            print(f"  loaded summary cache ({len(out):,} DOIs matched); skipping batch reads")
            return out, n_total, n_total

    for rnd in range(1, MAX_ROUNDS + 1):
        pending = [b for b in batches if not _scan_cache_path(b).exists()]
        if not pending or _budget_exhausted.is_set():
            break
        print(f"  round {rnd}: {len(pending):,}/{n_total:,} batches to fetch")
        failed = _run_round(pending)
        if _budget_exhausted.is_set():
            break
        if failed:
            wait = min(120, 30 * rnd)
            print(f"  round {rnd}: {failed:,} batches failed — pausing {wait}s before retry")
            time.sleep(wait)

    leftover = sum(1 for b in batches if not _scan_cache_path(b).exists())
    n_cached = n_total - leftover

    out: dict[str, bool] = {}
    for b in batches:
        cache = _scan_cache_path(b)
        if not cache.exists():
            continue
        for work in json.loads(cache.read_text()).get("results", []):
            doi = (work.get("doi") or "").replace("https://doi.org/", "").lower() or None
            if doi:
                out[doi] = bool(work.get("abstract_inverted_index"))

    print(f"  {n_cached:,}/{n_total:,} batches cached; matched {len(out):,}/{len(uniq):,} DOIs")

    # Persist the merged result so future runs bypass the batch-file scan.
    if n_cached == n_total:
        SCAN_SUMMARY.write_text(json.dumps(
            {"n_total": n_total, "n_dois": len(uniq), "scan": out},
            ensure_ascii=False,
        ))
        print(f"  wrote summary cache → {SCAN_SUMMARY}")

    return out, n_cached, n_total


def _aggregate(df: pd.DataFrame, scan: dict[str, bool]) -> pd.DataFrame:
    matched = set(scan)
    with_abstract = {d for d, has in scan.items() if has}

    df = df.assign(
        _has_doi=df["doi"].notna(),
        _oa_matched=df["doi"].isin(matched),
        _has_abstract=df["doi"].isin(with_abstract),
    )
    g = df.groupby("conf")
    table = pd.DataFrame({
        "n_total": g.size(),
        "n_doi": g["_has_doi"].sum(),
        "n_oa_matched": g["_oa_matched"].sum(),
        "n_abstract": g["_has_abstract"].sum(),
        "year_first": g["year"].min(),
        "year_last": g["year"].max(),
    }).reset_index()
    table["n_doi_less"] = table["n_total"] - table["n_doi"]
    table["abstract_hit_rate"] = (table["n_abstract"] / table["n_total"]).round(4)
    table["doi_coverage"] = (table["n_doi"] / table["n_total"]).round(4)
    table = table[[
        "conf", "n_total", "n_doi", "n_doi_less",
        "n_oa_matched", "n_abstract", "abstract_hit_rate", "doi_coverage",
        "year_first", "year_last",
    ]]
    return table.sort_values(
        ["abstract_hit_rate", "n_total"], ascending=[False, False]
    ).reset_index(drop=True)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([head, sep, *body])


def _write_report(df: pd.DataFrame, ranked: pd.DataFrame, scan: dict[str, bool]) -> None:
    n_papers = len(df)
    n_doi = int(df["doi"].notna().sum())
    n_abstract = int(df["doi"].isin({d for d, h in scan.items() if h}).sum())
    lines = [
        "# DBLP conference abstract hit rate",
        "",
        f"- Conference papers (inproceedings): **{n_papers:,}**",
        f"- Distinct venues: **{df['conf'].nunique():,}**",
        f"- Papers with a DOI: **{n_doi:,}** ({100 * n_doi / n_papers:.1f}%)",
        f"- Papers with an OpenAlex abstract: **{n_abstract:,}** "
        f"({100 * n_abstract / n_papers:.1f}%)",
        f"- Venues ranked below (>= {MIN_VENUE_SIZE} papers): **{len(ranked):,}**",
        "",
        "## Method",
        "",
        "Abstract availability comes from OpenAlex via DOI lookup only "
        "(per-paper title search does not scale to this corpus). A paper is a "
        "**hit** iff it has a DOI and OpenAlex carries an abstract for it; "
        "DOI-less papers are misses (dumped to `dblp_doi_less.parquet`). "
        "`abstract_hit_rate <= doi_coverage` always holds.",
        "",
        f"## Top 20 venues (>= {MIN_VENUE_SIZE} papers)",
        "",
        _md_table(ranked.head(20)),
        "",
        f"## Bottom 20 venues (>= {MIN_VENUE_SIZE} papers)",
        "",
        _md_table(ranked.tail(20)),
        "",
    ]
    dest = REPORTS_DIR / "conf_abstract_hit_rate.md"
    dest.write_text("\n".join(lines))
    print(f"Wrote {dest}")


def build_report() -> None:
    if not SRC.exists():
        raise SystemExit(f"Not found: {SRC}. Run `make dump` first.")

    df = pd.read_parquet(SRC)
    print(f"Loaded {len(df):,} conference papers from {SRC}")

    scan, n_cached, n_total = scan_dois(df["doi"].dropna().tolist())
    if n_cached < n_total:
        pct = 100 * n_cached / n_total
        raise SystemExit(
            f"\nScan incomplete: {n_cached:,}/{n_total:,} batches cached ({pct:.1f}%).\n"
            "OpenAlex's daily budget is likely exhausted (resets midnight UTC). "
            "Cached batches are saved — re-run `make conf-report PYTHON=.venv/bin/python` "
            "after the reset to continue. The report is written only once all batches are cached."
        )
    table = _aggregate(df, scan)

    all_path = TABLES_DIR / "conf_abstract_hit_rate_all.csv"
    table.to_csv(all_path, index=False)
    print(f"Wrote {all_path} ({len(table):,} venues)")

    ranked = table[table["n_total"] >= MIN_VENUE_SIZE].reset_index(drop=True)
    ranked_path = TABLES_DIR / "conf_abstract_hit_rate.csv"
    ranked.to_csv(ranked_path, index=False)
    print(f"Wrote {ranked_path} ({len(ranked):,} venues >= {MIN_VENUE_SIZE} papers)")

    doi_less = df[df["doi"].isna()][["conf", "dblp_key", "title", "year"]]
    doi_less_path = INTERIM_DIR / "dblp_doi_less.parquet"
    doi_less.to_parquet(doi_less_path, index=False)
    sample_path = TABLES_DIR / "dblp_doi_less_sample.csv"
    doi_less.head(SAMPLE_ROWS).to_csv(sample_path, index=False)
    print(f"Wrote {doi_less_path} ({len(doi_less):,} DOI-less papers) "
          f"and {sample_path} (first {min(SAMPLE_ROWS, len(doi_less)):,})")

    _write_report(df, ranked, scan)


if __name__ == "__main__":
    build_report()
