"""
ingest_dblp_dump.py — Download the full DBLP XML dump and flatten every
conference paper into one parquet.

Unlike ingest.py (which pulls a single venue per-year via the DBLP search API),
this fetches the complete dump so we can measure abstract availability across
*all* conferences. We only keep <inproceedings> (conference/workshop papers).

Reads:  https://dblp.org/xml/dblp.xml.gz  (+ dblp.dtd for entity resolution)
Writes: data/interim/dblp_conf.parquet

Columns: conf, dblp_key, title, year, doi, authors, ee, url

The per-venue path (topicdrift.ingest.venue) reads this same parquet and slices
it by `conf`, so we capture everything that downstream needs from one parse.

The dump uses custom HTML entities (&auml; etc.) declared in dblp.dtd, so a
plain xml.etree parse chokes. We stream-parse with lxml, which resolves those
entities from the adjacent dblp.dtd. Memory stays flat by clearing each element
and its already-seen siblings (the canonical dblp iterparse recipe).
"""

import gzip
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import requests
from lxml import etree

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

DBLP_GZ_URL = "https://dblp.org/xml/dblp.xml.gz"
DBLP_DTD_URL = "https://dblp.org/xml/dblp.dtd"
GZ_PATH = RAW_DIR / "dblp.xml.gz"
DTD_PATH = RAW_DIR / "dblp.dtd"
XML_PATH = RAW_DIR / "dblp.xml"  # decompressed; lives next to dblp.dtd
OUT_PATH = INTERIM_DIR / "dblp_conf.parquet"

# DBLP marks non-papers (corrigenda, datasets, etc.) with these publtype values.
SKIP_PUBLTYPES = {"withdrawn", "encyclopedia", "software", "data"}
DOI_RE = re.compile(r"doi\.org/(.+)$", re.I)

CHUNK = 1 << 20  # 1 MiB


def _download(url: str, dest: Path, max_attempts: int = 6) -> None:
    """Stream a URL to disk, skipping if cached. Resumes a partial download
    (HTTP Range) and retries with backoff, since the 1 GB .gz fetch is flaky."""
    if dest.exists():
        print(f"  [cache] {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    print(f"  [fetch] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(max_attempts):
        resume = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={resume}-"} if resume else {}
        try:
            with requests.get(
                url, stream=True, timeout=(30, 120), headers=headers
            ) as r:
                # 206 => server honored Range (append); else (re)start from scratch.
                append = resume and r.status_code == 206
                if not append:
                    r.raise_for_status()
                    resume = 0
                total = int(r.headers.get("content-length", 0)) + resume
                done = resume
                with open(tmp, "ab" if append else "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(
                                f"\r    {done / 1e9:.2f}/{total / 1e9:.2f} GB", end=""
                            )
            print()
            tmp.rename(dest)
            return
        except requests.RequestException as e:
            wait = min(60, 2**attempt * 5)
            print(
                f"\n    download error ({type(e).__name__}) — retrying in {wait}s "
                f"[{attempt + 1}/{max_attempts}]"
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {url} after {max_attempts} attempts")


def _decompress() -> None:
    """Inflate dblp.xml.gz next to dblp.dtd so lxml can resolve the DTD."""
    if XML_PATH.exists():
        print(f"  [cache] {XML_PATH.name} ({XML_PATH.stat().st_size / 1e9:.2f} GB)")
        return
    print(f"  [gunzip] {GZ_PATH.name} -> {XML_PATH.name}")
    tmp = XML_PATH.with_suffix(".xml.part")
    with gzip.open(GZ_PATH, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, CHUNK)
    tmp.rename(XML_PATH)


def _doi_and_ee(elem) -> tuple[str | None, str | None]:
    """Return (doi, first_ee_url). Walks <ee> once instead of twice."""
    first_ee = None
    doi = None
    for ee in elem.iterfind("ee"):
        text = (ee.text or "").strip()
        if not text:
            continue
        if first_ee is None:
            first_ee = text
        if doi is None:
            m = DOI_RE.search(text)
            if m:
                doi = m.group(1).strip().lower() or None
    return doi, first_ee


def _parse_inproceedings() -> pd.DataFrame:
    """Stream every <inproceedings>, flattening into rows."""
    rows = []
    kept = skipped = 0
    context = etree.iterparse(
        str(XML_PATH), events=("end",), tag="inproceedings", load_dtd=True
    )
    for _, elem in context:
        publtype = elem.get("publtype")
        key = elem.get("key") or ""
        if publtype in SKIP_PUBLTYPES or not key.startswith("conf/"):
            skipped += 1
        else:
            parts = key.split("/")
            title_el = elem.find("title")
            year_el = elem.find("year")
            url_el = elem.find("url")
            doi, ee = _doi_and_ee(elem)
            rows.append(
                {
                    "conf": "/".join(parts[:2]),  # conf/<slug>
                    "dblp_key": key,
                    "title": (
                        "".join(title_el.itertext()).strip()
                        if title_el is not None
                        else ""
                    ),
                    "year": (
                        int(year_el.text)
                        if year_el is not None and year_el.text
                        else None
                    ),
                    "doi": doi,
                    "authors": [
                        (a.text or "").strip()
                        for a in elem.iterfind("author")
                        if a.text
                    ],
                    "ee": ee,
                    "url": (
                        (url_el.text or "").strip()
                        if url_el is not None and url_el.text
                        else None
                    ),
                }
            )
            kept += 1

        # Keep memory flat: drop this element and earlier siblings under <dblp>.
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if (kept + skipped) % 500_000 == 0:
            print(f"  scanned {kept + skipped:,} inproceedings ({kept:,} kept)")

    print(f"  kept {kept:,} conference papers, skipped {skipped:,}")
    return pd.DataFrame(rows)


def build_dump() -> None:
    print("Downloading DBLP dump...")
    _download(DBLP_DTD_URL, DTD_PATH)
    _download(DBLP_GZ_URL, GZ_PATH)
    _decompress()

    print("Parsing inproceedings...")
    df = _parse_inproceedings()
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    df.to_parquet(OUT_PATH, index=False)
    n_conf = df["conf"].nunique()
    doi_cov = 100 * df["doi"].notna().mean()
    print(
        f"Wrote {OUT_PATH} ({len(df):,} papers, {n_conf:,} venues, {doi_cov:.1f}% with DOI)"
    )

    # Reclaim ~4.5 GB; keep the .gz so re-runs skip the download.
    XML_PATH.unlink(missing_ok=True)
    print(f"  removed {XML_PATH.name} (kept {GZ_PATH.name})")


if __name__ == "__main__":
    build_dump()
