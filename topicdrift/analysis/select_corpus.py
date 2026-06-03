"""
select_corpus.py — Pick the global-fit corpus from conf_enriched.parquet.

Multi-conference site pipeline, step 1/4:
  select_corpus -> topics_conf -> map_seed_themes -> apply_topic_groups

Writes conf_universe.parquet (dblp_key, conf, year, in_fit): every abstract-
bearing paper in a qualifying venue (the assignment universe), with in_fit
marking the stratified subset the clustering is fit on (scope venues in full,
the long tail capped per venue). Also writes outputs/tables/corpus_venues.csv.

Sampling only the fit bounds memory without changing which topics are found.
Thresholds + fit settings come from config/venues.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

PROCESSED_DIR = Path("data/processed")
OUTPUTS_TABLES = Path("outputs/tables")
OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)

CONF_ENRICHED = PROCESSED_DIR / "conf_enriched.parquet"
UNIVERSE_OUT = PROCESSED_DIR / "conf_universe.parquet"
VENUES_CFG = Path("config/venues.yaml")


def load_config() -> dict:
    return yaml.safe_load(VENUES_CFG.read_text())


def scope_venues(cfg: dict) -> set[str]:
    """Venues that must always be present in full (ICSE + Top-10)."""
    scopes = cfg.get("scopes", {})
    venues: set[str] = set(scopes.get("icse", []))
    venues.update(scopes.get("top10", []))
    return venues


def select() -> None:
    cfg = load_config()
    fit_cfg = cfg.get("fit", {})
    all_cfg = cfg.get("scopes", {}).get("all", {})
    min_papers = int(all_cfg.get("min_papers", 50))
    min_rate = float(all_cfg.get("min_abstract_rate", 0.5))
    sample_size = int(fit_cfg.get("sample_size", 250_000))
    per_venue_cap = int(fit_cfg.get("per_venue_cap", 4_000))
    seed = int(fit_cfg.get("seed", 42))
    forced = scope_venues(cfg)

    print(f"Reading {CONF_ENRICHED} (conf, dblp_key, year, has_abstract)…")
    df = pq.read_table(
        CONF_ENRICHED, columns=["conf", "dblp_key", "year", "has_abstract"]
    ).to_pandas()
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Per-venue stats over ALL papers (abstract rate is over the venue's papers).
    stats = df.groupby("conf").agg(
        n_papers=("dblp_key", "size"),
        n_abstract=("has_abstract", "sum"),
    )
    stats["abstract_rate"] = stats["n_abstract"] / stats["n_papers"]

    qualifies = (stats["n_papers"] >= min_papers) & (stats["abstract_rate"] >= min_rate)
    keep_venues = set(stats.index[qualifies]) | forced
    print(
        f"  {len(stats)} venues total; {int(qualifies.sum())} pass "
        f"(>= {min_papers} papers & >= {min_rate:.0%} abstracts); "
        f"+{len(forced - set(stats.index[qualifies]))} forced scope venues"
    )

    # Assignment universe = abstract-bearing papers in kept venues.
    uni = df[df["has_abstract"] & df["conf"].isin(keep_venues)][
        ["dblp_key", "conf", "year"]
    ].reset_index(drop=True)
    print(
        f"  universe: {len(uni):,} abstract-bearing papers across "
        f"{uni['conf'].nunique():,} venues"
    )

    # ── Stratified fit sample ────────────────────────────────────────────────
    is_scope = uni["conf"].isin(forced)
    scope_keys = uni.index[is_scope]  # always in the fit
    budget = max(0, sample_size - len(scope_keys))

    # Tail: cap each venue, then scale down proportionally to fit the budget.
    tail = uni[~is_scope]
    per_venue = tail.groupby("conf").size().clip(upper=per_venue_cap)
    planned = int(per_venue.sum())
    if planned > budget and planned > 0:
        scale = budget / planned
        per_venue = (per_venue * scale).round().astype(int).clip(lower=1)

    tail_by_venue = {c: g.index for c, g in tail.groupby("conf")}
    tail_picks: list[int] = []
    for conf, k in per_venue.items():
        rows = tail_by_venue.get(conf)
        if rows is None:
            continue
        k = min(int(k), len(rows))
        if k > 0:
            tail_picks.extend(rows.to_series().sample(n=k, random_state=seed).tolist())
    tail_idx = pd.Index(tail_picks)

    fit_idx = scope_keys.union(tail_idx)
    uni["in_fit"] = False
    uni.loc[fit_idx, "in_fit"] = True

    uni.to_parquet(UNIVERSE_OUT, index=False)
    print(
        f"  fit sample: {int(uni['in_fit'].sum()):,} docs "
        f"({len(scope_keys):,} scope + {len(tail_idx):,} tail)"
    )
    print(f"  wrote {UNIVERSE_OUT}")

    # ── Per-venue report ─────────────────────────────────────────────────────
    fit_per_venue = uni[uni["in_fit"]].groupby("conf").size().rename("n_fit")
    report = (
        stats.assign(
            kept=stats.index.isin(keep_venues),
            n_universe=uni.groupby("conf")
            .size()
            .reindex(stats.index)
            .fillna(0)
            .astype(int),
            n_fit=fit_per_venue.reindex(stats.index).fillna(0).astype(int),
        )
        .sort_values("n_universe", ascending=False)
        .reset_index()
    )
    report.to_csv(OUTPUTS_TABLES / "corpus_venues.csv", index=False)
    print(f"  wrote {OUTPUTS_TABLES / 'corpus_venues.csv'} ({len(report):,} venues)")


if __name__ == "__main__":
    select()
