"""
apply_topic_groups.py — Stamp the overarching grouping into the data.

Multi-conference site pipeline, step 4/4:
  select_corpus -> topics_conf -> map_seed_themes -> apply_topic_groups

Reads a grouping YAML (the source of truth) and, for the given file prefix:
  • adds a `group` column to {prefix}topics/paper_topics(/topics_over_time),
  • writes the group registry {prefix}topic_groups.parquet (+ .csv), and
  • regenerates the {prefix}topic_groups.md report from the data.

Edit the YAML and re-run to re-group — no re-fit needed. Defaults target the
global fit; pass --prefix/--config for another set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path("config/topic_groups.conf.yaml")
PROCESSED_DIR = Path("data/processed")
OUTPUTS_TABLES = Path("outputs/tables")
OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────


def load_groups(path: Path = CONFIG_PATH) -> list[dict]:
    """Ordered list of {name, color, topics:[label, ...]} from the YAML."""
    cfg = yaml.safe_load(path.read_text())
    groups = cfg.get("groups") if isinstance(cfg, dict) else None
    if not groups:
        raise ValueError(f"{path} has no `groups:` list")
    return groups


def build_label_map(
    groups: list[dict],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Return ({label: group}, {group: color}, [group order])."""
    label_to_group: dict[str, str] = {}
    group_color: dict[str, str] = {}
    order: list[str] = []
    for g in groups:
        name = str(g["name"]).strip()
        order.append(name)
        group_color[name] = str(g.get("color", "#94a3b8"))
        for label in g.get("topics", []):
            label = str(label).strip()
            if label in label_to_group:
                raise ValueError(
                    f"label {label!r} listed twice ({label_to_group[label]!r} and {name!r})"
                )
            label_to_group[label] = name
    return label_to_group, group_color, order


def validate(
    label_to_group: dict[str, str],
    topic_labels: set[str],
    config_name: str = "the grouping YAML",
) -> None:
    """Fail loudly if the YAML and the fitted topics disagree."""
    config_labels = set(label_to_group)
    missing = topic_labels - config_labels  # fitted but ungrouped
    unknown = config_labels - topic_labels  # grouped but not fitted
    problems = []
    if missing:
        problems.append(
            f"topics in the data with no group in {config_name}:\n    "
            + "\n    ".join(sorted(missing))
        )
    if unknown:
        problems.append(
            f"labels in {config_name} not present in the topics table "
            "(stale after a re-fit?):\n    " + "\n    ".join(sorted(unknown))
        )
    if problems:
        raise SystemExit(
            f"{config_name} is out of sync with the data:\n\n"
            + "\n\n".join(problems)
            + f"\n\nEdit {config_name} so every fitted topic is grouped "
            "exactly once, then re-run."
        )


# ── Apply ───────────────────────────────────────────────────────────────────


def stamp_group_column(df: pd.DataFrame, id_to_group: dict[int, str]) -> pd.DataFrame:
    """Add/replace a `group` column from topic_id, idempotently."""
    df = df.drop(columns=[c for c in ("group",) if c in df.columns])
    df["group"] = df["topic_id"].astype(int).map(id_to_group)
    return df


def write_registry(
    order: list[str],
    group_color: dict[str, str],
    group_papers: dict[str, int],
    total: int,
    prefix: str,
) -> pd.DataFrame:
    """Canonical group-level table consumed by the visualizations."""
    rows = [
        {
            "group": g,
            "order": i,
            "color": group_color[g],
            "papers": int(group_papers.get(g, 0)),
            "share": round(group_papers.get(g, 0) / total, 6) if total else 0.0,
        }
        for i, g in enumerate(order)
    ]
    reg = pd.DataFrame(rows)
    reg.to_parquet(PROCESSED_DIR / f"{prefix}topic_groups.parquet", index=False)
    reg.to_csv(OUTPUTS_TABLES / f"{prefix}topic_groups.csv", index=False)
    print(f"  wrote {prefix}topic_groups.parquet + {prefix}topic_groups.csv ({len(reg)} groups)")
    return reg


def render_md(
    groups: list[dict], topics: pd.DataFrame, total: int, title: str, config_name: str
) -> str:
    """Regenerate the topic-groups report from the data (counts live here)."""
    size_by_label = dict(zip(topics["llm_label"], topics["size"].astype(int)))
    n_topics = len(topics)
    n_groups = len(groups)

    def pct(n: int) -> str:
        return f"{100 * n / total:.1f}%" if total else "0.0%"

    # Group totals + display order (by paper volume, desc — matches the chart).
    group_total = {g["name"]: sum(size_by_label.get(t, 0) for t in g["topics"]) for g in groups}
    ordered = sorted(groups, key=lambda g: group_total[g["name"]], reverse=True)

    lines: list[str] = []
    lines.append(f"# {title} topic groups")
    lines.append("")
    lines.append(
        f"{n_topics} BERTopic topics, labelled by a local instruct LLM "
        f"(Qwen2.5-3B-Instruct), grouped into {n_groups}"
    )
    lines.append(
        f"overarching themes per {config_name}. Percent = share of the "
        f"{total:,} grouped papers with abstracts."
    )
    lines.append("")
    lines.append("Generated by src/analysis/apply_topic_groups.py — do not edit by hand.")
    lines.append("")
    lines.append("| # | Theme | Papers | % |")
    lines.append("|---:|---|---:|---:|")
    for i, g in enumerate(ordered, 1):
        n = group_total[g["name"]]
        lines.append(f"| {i} | {g['name']} | {n:,} | {pct(n)} |")
    lines.append(f"| | **Total** | **{total:,}** | **100%** |")
    lines.append("")

    for i, g in enumerate(ordered, 1):
        n = group_total[g["name"]]
        lines.append("---")
        lines.append("")
        lines.append(f"## Group {i} — {g['name']} · {pct(n)}")
        lines.append("")
        lines.append("| Topic | Papers | % |")
        lines.append("|---|---:|---:|")
        members = sorted(g["topics"], key=lambda t: size_by_label.get(t, 0), reverse=True)
        for t in members:
            sz = size_by_label.get(t, 0)
            lines.append(f"| {t} | {sz:,} | {pct(sz)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(
    prefix: str = "conf_",
    config_path: Path = CONFIG_PATH,
    title: str = "All Conferences",
) -> None:
    groups = load_groups(config_path)
    label_to_group, group_color, order = build_label_map(groups)

    topics = pd.read_parquet(PROCESSED_DIR / f"{prefix}topics.parquet")
    fitted = topics[topics["topic_id"] != -1].copy()
    validate(label_to_group, set(fitted["llm_label"].astype(str)), config_path.name)

    id_to_group = {
        int(r["topic_id"]): label_to_group[str(r["llm_label"])] for _, r in fitted.iterrows()
    }
    total = int(fitted["size"].sum())

    # 1) {prefix}topics.parquet — add group (+ color for convenience)
    topics_out = stamp_group_column(topics, id_to_group)
    topics_out = topics_out.drop(columns=[c for c in ("group_color",) if c in topics_out.columns])
    topics_out["group_color"] = topics_out["group"].map(group_color)
    topics_out.to_parquet(PROCESSED_DIR / f"{prefix}topics.parquet", index=False)
    print(
        f"  stamped group into {prefix}topics.parquet "
        f"({topics_out['group'].notna().sum()}/{len(topics_out)} rows)"
    )

    # 2) {prefix}topics_over_time.parquet (optional)
    tot_path = PROCESSED_DIR / f"{prefix}topics_over_time.parquet"
    if tot_path.exists():
        tot = stamp_group_column(pd.read_parquet(tot_path), id_to_group)
        tot.to_parquet(tot_path, index=False)
        print(f"  stamped group into {prefix}topics_over_time.parquet ({len(tot)} rows)")

    # 3) {prefix}paper_topics.parquet
    pt_path = PROCESSED_DIR / f"{prefix}paper_topics.parquet"
    if pt_path.exists():
        pt = stamp_group_column(pd.read_parquet(pt_path), id_to_group)
        pt.to_parquet(pt_path, index=False)
        print(f"  stamped group into {prefix}paper_topics.parquet ({len(pt)} rows)")

    # 4) group registry + 5) report
    group_papers = {
        g: int(fitted[fitted["topic_id"].map(id_to_group) == g]["size"].sum()) for g in order
    }
    write_registry(order, group_color, group_papers, total, prefix)

    md = render_md(groups, fitted, total, title, config_path.name)
    (OUTPUTS_TABLES / f"{prefix}topic_groups.md").write_text(md)
    print(f"  regenerated {prefix}topic_groups.md ({len(order)} groups, {total:,} papers)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="conf_", help="data file prefix, e.g. conf_ or icse_")
    ap.add_argument("--config", default=str(CONFIG_PATH), help="grouping YAML (source of truth)")
    ap.add_argument("--title", default="All Conferences", help="report title")
    a = ap.parse_args()
    main(prefix=a.prefix, config_path=Path(a.config), title=a.title)
