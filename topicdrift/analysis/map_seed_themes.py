"""
map_seed_themes.py — Fit the global topics into the original 10 ICSE themes.

Multi-conference site pipeline, step 3/4:
  select_corpus -> topics_conf -> map_seed_themes -> apply_topic_groups

Each seed theme is anchored by its original ICSE sub-topics; every global topic
is assigned to its nearest anchor (cosine on the topic centroid vs the embedded
anchor text).

Writes config/topic_groups.conf.proposed.yaml — a *proposed* mapping for human
review. The locked file at config/topic_groups.conf.yaml is the source of truth
consumed by apply_topic_groups.py and is intentionally NOT overwritten here, so
LLM-label drift across re-fits doesn't churn the live grouping. Diff the
proposed file against the locked one and copy changes in by hand when you want
them to take effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from topicdrift.topic_model import TopicModel

PROCESSED_DIR = Path("data/processed")
LOCKED_YAML = Path("config/topic_groups.conf.yaml")
OUT_YAML = Path("config/topic_groups.conf.proposed.yaml")

# Original ICSE themes → (colour, anchor sub-topics). Anchors seed the meaning.
SEEDS: dict[str, tuple[str, list[str]]] = {
    "Program Correctness": (
        "#2563eb",
        [
            "Temporal Logic Verification",
            "Pointer Analysis",
            "Specification",
            "Secure Development Lifecycle",
            "Dependency Conflict Detection",
            "Concurrency Bugs",
            "Contract Audit",
            "Certified Code Assurance",
            "Static Analysis",
            "Model Checking",
            "Formal Methods",
            "Security Privacy",
        ],
    ),
    "Human Factors in Software Engineering": (
        "#16a34a",
        [
            "Curriculum Design Education",
            "Software Diversity",
            "Interactive Design",
            "Emotion Coding",
            "Gender Bias",
            "Collaborative Development",
            "Accessibility",
            "Human Computer Interaction",
            "Ethics Society",
        ],
    ),
    "System Design": (
        "#9333ea",
        [
            "Software Architecture",
            "Interface Decomposition",
            "Feature Variability",
            "Uml Modeling",
            "Crosscutting Concerns",
            "Software Reuse",
            "Service Oriented Architecture",
            "Microservices",
            "Self Adaptive Systems",
            "Digital Twin",
            "Distributed Software",
            "Parallel Concurrent Computing",
        ],
    ),
    "Developer Tooling": (
        "#d97706",
        [
            "Automated Code Review",
            "Code Navigation",
            "Open Source",
            "Code Search",
            "Code Synthesis",
            "Refactoring",
            "Clone Detection",
            "Program Comprehension",
            "Software Evolution",
            "Dependency Analysis",
        ],
    ),
    "Software Testing": (
        "#0891b2",
        [
            "Testing",
            "Fuzzing",
            "Software Reliability",
            "Code Coverage",
            "Web Gui Testing",
            "Metamorphic Testing",
            "Test Generation",
        ],
    ),
    "Emerging Platforms": (
        "#e11d48",
        [
            "Mobile Android Apps",
            "Iot Internet Of Things",
            "Cloud Edge Computing",
            "Web Javascript",
            "Blockchain",
            "Robotics Manipulation",
            "Wireless Networks",
            "Antenna Circuit Hardware",
            "Signal Processing",
            "Radar",
            "Control Systems",
            "Optical Networks",
            "Fpga Embedded Hardware",
            "Low Power Electronics",
            "Autonomous Vehicles",
            "Smart City Grid",
            "Sensor Networks",
            "Channel Estimation",
        ],
    ),
    "Defect Management": (
        "#ea580c",
        [
            "Automated Program Repair",
            "Fault Classification",
            "Defect Prediction",
            "Bug Localization",
            "Log Analysis",
            "Anomaly Detection",
            "Continuous Inspection",
        ],
    ),
    "Software Process": (
        "#4f46e5",
        [
            "Process Integration",
            "Software Quality Assurance",
            "Agile Methodologies",
            "Effort Prediction",
            "Software Analytics",
            "Project Management",
            "Technical Debt",
            "Maintenance",
        ],
    ),
    "AI for Software Engineering": (
        "#0d9488",
        [
            "Machine Learning",
            "Deep Learning Neural Network",
            "Natural Language Processing",
            "Computer Vision Image",
            "Speech Recognition",
            "Data Mining",
            "Recommendation",
            "Predictive Analytics",
            "Graph Neural Network",
        ],
    ),
    "Requirements Engineering": (
        "#65a30d",
        [
            "Requirement Alignment",
            "Semantic Integration",
            "Goal Oriented Design",
            "Traceability",
            "Requirements Specification",
        ],
    ),
}


def _dedupe(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        out.append(f"{lab} ({seen[lab]})" if seen[lab] > 1 else lab)
    return out


def main() -> None:
    topics = pd.read_parquet(PROCESSED_DIR / "conf_topics.parquet")
    topics["llm_label"] = _dedupe(topics["llm_label"].astype(str).tolist())
    topics.to_parquet(PROCESSED_DIR / "conf_topics.parquet", index=False)
    centroids = np.load(PROCESSED_DIR / "conf_topic_centroids.npy")
    ordered_ids = sorted(topics["topic_id"].astype(int).tolist())
    label_of = dict(zip(topics["topic_id"].astype(int), topics["llm_label"].astype(str)))

    embedder = TopicModel().embedder()
    anchor_text = [f"{name}. " + ", ".join(words) for name, (_, words) in SEEDS.items()]
    anchors = np.asarray(embedder.encode(anchor_text, normalize_embeddings=True), dtype=np.float32)
    theme_names = list(SEEDS)

    sims = centroids @ anchors.T  # (n_topics, 10)
    best = sims.argmax(axis=1)

    members: dict[str, list[tuple[str, int]]] = {n: [] for n in theme_names}
    size = dict(zip(topics["llm_label"], topics["size"].astype(int)))
    for row, tid in enumerate(ordered_ids):
        lab = label_of[tid]
        members[theme_names[best[row]]].append((lab, int(size.get(lab, 0))))

    groups = []
    for name in theme_names:
        mem = sorted(members[name], key=lambda x: -x[1])
        if not mem:
            continue
        groups.append({"name": name, "color": SEEDS[name][0], "topics": [m[0] for m in mem]})
    groups.sort(key=lambda g: sum(size.get(t, 0) for t in g["topics"]), reverse=True)

    header = (
        f"# {OUT_YAML.name} — PROPOSED mapping from a fresh topic fit.\n"
        f"# Source of truth is {LOCKED_YAML.name} (locked, hand-edited).\n"
        "# Review this file, then copy desired changes into the locked one.\n"
        "# Diff:  diff -u config/topic_groups.conf.yaml config/topic_groups.conf.proposed.yaml\n\n"
    )
    OUT_YAML.write_text(
        header + yaml.safe_dump({"groups": groups}, sort_keys=False, allow_unicode=True, width=100)
    )
    print(f"  wrote {OUT_YAML} ({len(groups)} themes over {len(ordered_ids)} topics)")
    if LOCKED_YAML.exists():
        print(f"  (locked file {LOCKED_YAML} unchanged — diff to review)")
    else:
        print(f"  NOTE: {LOCKED_YAML} does not exist; copy the proposed file to lock it in")
    for g in groups:
        print(f"    {g['name']:<40} {len(g['topics'])} topics")


if __name__ == "__main__":
    main()
