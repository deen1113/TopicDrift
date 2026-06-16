# TopicDrift: Findings at a Glance

A plain-language summary of our findings with this project. You can read more below.
**Explore it live:** https://deen1113.github.io/TopicDrift/icse.html

---

## Major findings

- **ICSE changed what it cares about.** Over five decades it moved away from
  *architecture and process* research toward *code-level tooling, repository
  mining, testing, and AI-assisted engineering*.
- **The biggest drops:** System Design fell from 28% to 10% of papers, and
  Software Process more than halved (15% to 6%).
- **The biggest rises:** Developer Tooling rose from 13% to 22% and Defect
  Management from 4% to 12%, with Testing and AI for SE climbing alongside.
- **The shift sped up around 2015 to 2020**, the years the tooling and AI topics
  surged, then began to settle.
- **Volume isn't influence.** Defect Management punches above its weight (12% of
  papers but 17% of citations); *Defect Prediction* papers average about 95
  citations. Human Factors and Emerging Platforms appear steadily but are cited
  little.
- **ICSE is more of a software engineering venue, not an AI one.** AI is just 6.9% of
  ICSE papers but 36% of all computer science, a contrast only visible because we
  use one shared topic model across every venue.

---

## Charts that illustrate the results

| Chart | What it shows | Link |
|---|---|---|
| **Theme Drift** (streamgraph) | Each theme's share of ICSE papers over time; the core "centre of gravity" shift | [open](https://deen1113.github.io/TopicDrift/visualizations/topic_group_streamgraph_icse.html) |
| **Theme Rankings** (bump chart) | The shifting pecking order, where one theme overtakes another | [open](https://deen1113.github.io/TopicDrift/visualizations/theme_rank_bump.html) |
| **Corpus Composition** (treemap) | Volume by decade and topic; recolour by citation impact | [open](https://deen1113.github.io/TopicDrift/visualizations/topic_treemap_icse.html) |
| **Topic Drift Search** | Search any keyword and watch its topic rise and fall | [open](https://deen1113.github.io/TopicDrift/visualizations/topic_drift_search.html) |
| **Researcher Migration** | How researchers move between themes over time | [open](https://deen1113.github.io/TopicDrift/visualizations/researcher_migration_sankey.html) |
| **Vocabulary Turnover** | How much a theme's defining words change each window | [open](https://deen1113.github.io/TopicDrift/visualizations/lexical_turnover.html) |

*All charts are interactive: hover, click, and toggle.*

---

## The numbers behind it

**ICSE then vs. now (theme share of papers):**

| Theme | Early (<=2005) | Recent (>=2016) |
|---|---:|---:|
| System Design | 28.1% | 9.7% |
| Software Process | 14.9% | 5.8% |
| Developer Tooling | 13.0% | 21.7% |
| Defect Management | 3.9% | 12.4% |
| Software Testing | 4.6% | 10.2% |
| AI for Software Engineering | 4.3% | 8.7% |

**ICSE vs. the wider field (share of papers):**

| Theme | ICSE | Top-10 venues | All of CS |
|---|---:|---:|---:|
| AI for Software Engineering | 6.9% | 5.7% | **36.1%** |
| Developer Tooling | 18.4% | 19.8% | 3.5% |
| Software Testing | 8.8% | 7.6% | 0.9% |

---

## Where it falls short (read the findings with this in mind)

We're upfront that the topic and theme layer is the weakest part of the work,
and we measured how weak:

- **The 10 themes don't separate cleanly.** Our grouping of 174 topics into 10
  themes is a hand-made judgment call, not a validated partition, and the numbers
  show it: about 56% of topics sit closer to a *different* theme than the one we
  filed them under. Some could reasonably belong to more than one theme, and a
  few are arguably mislabelled.
- **Topics themselves are only loosely separated.** They're distinct in
  vocabulary but blur together in meaning, so individual topic boundaries
  shouldn't be read too literally.
- **Ingestion pulls in more than it should.** The corpus still includes
  non-software-engineering (and some non-CS) work that leaks in during data
  collection, which inflates and smears the broader themes. Better filtering and
  cleaning is the most important next step.

In short: the direction of the findings is somewhat reliable; the precision of the
topic/theme labels is a known limitation we'd improve with better ingestion
filtering and a more principled, data-driven grouping.
