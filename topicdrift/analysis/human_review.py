"""
Interactive human review of topic/group classification accuracy.

Samples papers, presents title + abstract, asks for correct/incorrect
judgment, and saves results incrementally so the session is resumable.

Usage:
    make human-review                          # ICSE, n=100, seed=42
    make human-review CONF=conf/icse
    python -m topicdrift.analysis.human_review --results-only
    python -m topicdrift.analysis.human_review --rereview-incorrect
    python -m topicdrift.analysis.human_review --conf conf/icse --n 50
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
INTERIM = ROOT / "data/interim"
RESULTS = PROCESSED / "human_review_results.csv"

GROUPS = [
    "AI for Software Engineering",
    "System Design",
    "Emerging Platforms",
    "Program Correctness",
    "Human Factors in Software Engineering",
    "Developer Tooling",
    "Defect Management",
    "Requirements Engineering",
    "Software Process",
    "Software Testing",
]

FIELDS = ["dblp_key", "title", "group", "llm_label", "correct", "error_type", "notes"]
ERROR_LABELS = {"g": "wrong group", "t": "wrong topic", "b": "both wrong"}


def load_sample(n: int, seed: int, conf: str | None = None) -> pd.DataFrame:
    enriched = pd.read_parquet(
        INTERIM / "conf_enriched.parquet", columns=["dblp_key", "title", "abstract", "year", "doi"]
    )
    pt = pd.read_parquet(
        PROCESSED / "conf_paper_topics.parquet", columns=["dblp_key", "conf", "topic_id", "group"]
    )
    topics = pd.read_parquet(
        PROCESSED / "conf_topics.parquet", columns=["topic_id", "llm_label", "top_words"]
    )
    df = pt.merge(enriched, on="dblp_key", how="inner").merge(topics, on="topic_id", how="left")
    df = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]
    if conf:
        df = df[df["conf"] == conf]
        if df.empty:
            raise SystemExit(f"No papers found for conf={conf!r}")
    per_group = max(1, n // len(GROUPS))
    rng = np.random.default_rng(seed)
    frames = []
    for group in GROUPS:
        subset = df[df["group"] == group]
        k = min(per_group, len(subset))
        if k:
            frames.append(subset.sample(n=k, random_state=int(rng.integers(0, 2**31))))
    return pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)


def already_reviewed() -> set[str]:
    return set(pd.read_csv(RESULTS)["dblp_key"].tolist()) if RESULTS.exists() else set()


def append_result(row: dict) -> None:
    write_header = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def print_summary() -> None:
    if not RESULTS.exists():
        print("No results file found.")
        return
    df = pd.read_csv(RESULTS)
    correct = (df["correct"] == "y").sum()
    incorrect = (df["correct"] == "n").sum()
    skipped = (df["correct"] == "s").sum()
    reviewed = correct + incorrect
    print(f"\n{'=' * 60}")
    print(
        f"  Total : {len(df)}   Correct : {correct}   Incorrect : {incorrect}   Skipped : {skipped}"
    )
    if reviewed:
        print(f"  Accuracy (excl. skipped): {correct}/{reviewed} = {100 * correct / reviewed:.1f}%")
    print(f"{'=' * 60}\n")
    if incorrect:
        et = df.loc[df["correct"] == "n", "error_type"].fillna("")
        for code, label in ERROR_LABELS.items():
            print(f"  {label:<20}: {(et == code).sum()}")
        print()
        print("Incorrect papers:")
        for _, r in df[df["correct"] == "n"].iterrows():
            tag = f"[{r['error_type'].upper()}]" if str(r.get("error_type", "")).strip() else "   "
            print(f"  {tag} [{r['group']} / {r['llm_label']}]  {r['title'][:70]}")
            if str(r.get("notes", "")).strip():
                print(f"       Note: {r['notes']}")
    print("\nBreakdown by group:")
    for group in GROUPS:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        c = (sub["correct"] == "y").sum()
        n = (sub["correct"] != "s").sum()
        print(f"  {group:<45} {c:>2}/{n:<2}  [{'#' * c}{'.' * (n - c)}]")


def _wrap(text: str, width: int = 100, max_lines: int = 6) -> str:
    lines = textwrap.wrap(str(text), width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 3] + "..."
    return "\n  ".join(lines)


def display_paper(idx: int, total: int, row, header_extra: str = "") -> None:
    words = getattr(row, "top_words", [])
    kw = ", ".join(str(w) for w in (words[:8] if hasattr(words, "__len__") else []))
    year = getattr(row, "year", None)
    doi = getattr(row, "doi", None)
    print(f"\n{'─' * 60}")
    print(f"  [{idx}/{total}]{header_extra}")
    print(
        f"\n  TITLE:   {getattr(row, 'title', '')}"
        + (f" ({int(year)})" if year and not pd.isna(year) else "")
    )
    if doi and str(doi).strip():
        print(f"  DOI:     https://doi.org/{doi}")
    print(f"\n  ABSTRACT:\n  {_wrap(getattr(row, 'abstract', ''))}")
    print(f"\n  TOPIC:   {getattr(row, 'llm_label', '')}  (keywords: {kw})")
    print(f"  GROUP:   {getattr(row, 'group', '')}")
    print(f"{'─' * 60}")


def prompt_user(idx: int, total: int, row: pd.Series) -> tuple[str, str, str]:
    display_paper(idx, total, row)
    while True:
        ans = input("  Correct? [y]es / [n]o / [s]kip / [q]uit : ").strip().lower()
        if ans in ("y", "n", "s", "q"):
            break
        print("  Please enter y, n, s, or q.")
    error_type = notes = ""
    if ans == "n":
        while True:
            error_type = input("  What's wrong? [g]roup / [t]opic / [b]oth : ").strip().lower()
            if error_type in ("g", "t", "b"):
                break
            print("  Please enter g, t, or b.")
        notes = input("  Notes (optional, press Enter to skip): ").strip()
    return ans, error_type, notes


def rereview_incorrect() -> None:
    if not RESULTS.exists():
        print("No results file found.")
        return
    df = pd.read_csv(RESULTS)
    wrong = df[df["correct"] == "n"].copy()
    if wrong.empty:
        print("No incorrect papers to re-review.")
        return
    enriched = pd.read_parquet(
        INTERIM / "conf_enriched.parquet", columns=["dblp_key", "abstract", "year", "doi"]
    )
    pt = pd.read_parquet(PROCESSED / "conf_paper_topics.parquet", columns=["dblp_key", "topic_id"])
    topics = pd.read_parquet(PROCESSED / "conf_topics.parquet", columns=["topic_id", "top_words"])
    paper_data = (
        wrong[["dblp_key", "title", "group", "llm_label", "error_type", "notes"]]
        .merge(enriched, on="dblp_key", how="left")
        .merge(pt, on="dblp_key", how="left")
        .merge(topics, on="topic_id", how="left")
        .reset_index(drop=True)
    )
    updates: dict[str, tuple[str, str]] = {}
    for i, row in enumerate(paper_data.itertuples(index=False)):
        current = ERROR_LABELS.get(str(getattr(row, "error_type", "") or "").strip(), "not set")
        try:
            display_paper(
                i + 1,
                len(paper_data),
                row,
                header_extra=f"  *** INCORRECT — currently: {current} ***",
            )
            while True:
                et = input("  [g]roup / [t]opic / [b]oth / [s]kip / [q]uit : ").strip().lower()
                if et in ("g", "t", "b", "s", "q"):
                    break
            if et == "q":
                break
            if et == "s":
                continue
            note = input("  Notes (Enter to keep existing): ").strip()
            updates[getattr(row, "dblp_key")] = (et, note)
        except (KeyboardInterrupt, EOFError):
            print("\nInterrupted — saving progress.")
            break
    if updates:
        for key, (et, note) in updates.items():
            mask = df["dblp_key"] == key
            df.loc[mask, "error_type"] = et
            if note:
                df.loc[mask, "notes"] = note
        df.to_csv(RESULTS, index=False)
        print(f"\nUpdated {len(updates)} paper(s).")
    print_summary()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=str, default=None, help="e.g. conf/icse")
    ap.add_argument("--results-only", action="store_true")
    ap.add_argument("--rereview-incorrect", action="store_true")
    args = ap.parse_args()

    global RESULTS
    if args.conf:
        RESULTS = PROCESSED / f"human_review_results_{args.conf.replace('/', '_')}.csv"

    if args.results_only:
        print_summary()
        return
    if args.rereview_incorrect:
        rereview_incorrect()
        return

    print("Loading data...", end=" ", flush=True)
    sample = load_sample(args.n, args.seed, conf=args.conf)
    reviewed = already_reviewed()
    print(f"done. {len(sample)} papers sampled, {len(reviewed)} already reviewed.")

    remaining = sample[~sample["dblp_key"].isin(reviewed)].reset_index(drop=True)
    if remaining.empty:
        print("All sampled papers already reviewed.")
        print_summary()
        return

    print(
        f"  {len(remaining)} to review. Ctrl-C or 'q' to stop early.\n"
        "  Rate whether the GROUP assignment is correct, not just the topic label."
    )

    for i, row in enumerate(remaining.itertuples(index=False)):
        try:
            ans, error_type, notes = prompt_user(i + 1, len(remaining), row)
        except (KeyboardInterrupt, EOFError):
            print("\nInterrupted — saving progress.")
            break
        if ans == "q":
            print("Quitting — saving progress.")
            break
        append_result(
            {
                "dblp_key": getattr(row, "dblp_key", ""),
                "title": getattr(row, "title", ""),
                "group": getattr(row, "group", ""),
                "llm_label": getattr(row, "llm_label", ""),
                "correct": ans,
                "error_type": error_type,
                "notes": notes,
            }
        )
    print_summary()


if __name__ == "__main__":
    main()
