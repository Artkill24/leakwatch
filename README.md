# leakwatch

Audits deployed ML models for **target leakage** using catalog metadata alone —
no access to training data, no model artifacts, no code.

It reads what a model's features claim to do, checks those claims against the
temporal boundary the label imposes, and then walks column-level lineage to find
the cases where the claims are honest and the problem lies elsewhere.

---

## The problem

Target leakage is expensive and quiet. A feature encodes something knowable only
*after* the prediction point, the model scores beautifully in validation, and it
degrades the moment it meets production. By the time anyone notices, it has been
making decisions for months.

Catalogs already hold what is needed to catch this: what the label is computed
from, what each feature reads, and how the windows are anchored. Nobody reads it,
because reading it across dozens of models is tedious and nobody has the time.

That is a job for an agent.

---

## Three layers of analysis

Each layer catches something the one before it cannot.

| layer | question | who answers |
|-------|----------|-------------|
| **textual** | what does this description actually state? | LLM, extraction only |
| **temporal** | does a stated window cross the prediction boundary? | rules, in Python |
| **topological** | does this feature read the label's column under another name? | column-level lineage |

The third layer is what justifies having a catalog at all. Two features in the
seeded scenario are problematic, and different layers catch them:

| feature | reads as | caught by | why |
|---------|----------|-----------|-----|
| `post_cancellation_refund_total` | openly unbounded | textual + temporal | states it aggregates over the full snapshot, and the label is defined on a column of that same table |
| `fulfilment_exception_rate` | entirely clean | topological only | 90-day window, backward, anchored to the reference date, closed before scoring — but sourced from the raw table the label's column comes from |

For the second, the extractor reports `target_col=no`. The model read the
description and found nothing, correctly. The finding comes from the graph:

```
fulfilment_exception_rate ──sources──► analytics.order_details
analytics.order_details.order_status ──1 hop──► analytics.order_history.order_status
label is defined on ───────────────────────────► analytics.order_history.order_status
```

This is the raw-versus-curated trap. The label is defined on the curated table,
someone builds a feature from the raw one believing it unrelated, and no
description anywhere betrays it. Reading documentation cannot find this. Walking
the graph can.

### Lineage crosses platforms, not just tables

Resolving the ancestry of `order_history.order_status` returns:

```
1 hop  [snowflake] analytics.order_details.order_status
1 hop  [dbt      ] analytics.order_history.order_status
2 hop  [snowflake] order_entry.orders.order_status
2 hop  [dbt      ] analytics.order_details.order_status
3 hop  [dbt      ] order_entry.orders.order_status
```

The same logical table appears once per platform: a dbt model and the warehouse
table it produces are distinct catalog entities, and both genuinely carry the
column. A single traversal therefore spans the transformation layer and the
warehouse — which is the difference between using a catalog and reading a schema
dump.

---

## How it works

```
DataHub graph                    deterministic traversal, no LLM
  mlModel
    └─ mlFeature ───────────────► context.py   model → features → datasets → columns
         └─ dataset
              └─ schemaField ───► lineage.py   column ancestry of the label,
                                               cycle-safe, platform-aware
                                            │
                                            ▼
                                     temporal.py
                                       LLM extracts stated facts only
                                       ├─ window_length
                                       ├─ direction       backward|forward|unstated
                                       ├─ anchor          reference_date|other|unstated
                                       ├─ closes_before_scoring
                                       └─ mentions_target_column
                                            +
                                       reads_target_ancestor  ← from the graph
                                            │
                                            ▼
                                     rules in Python
                                       no model past this line
                                            │
                                            ▼
                                     agent.py → write-back
                                       tags + audit trail on the model
```

**The graph is walked in code, not by the LLM.** Handing an agent a set of MCP
tools and letting it wander costs a round trip per hop, varies between runs, and
cannot be tested. Traversal is a solved problem; judgement is what deserves the
tokens.

**The LLM reads, Python decides.** The model is asked what a description
*states*, never whether something is wrong. Every verdict comes from rules that
can be read and tested without an API key.

---

## Reliability

Most submissions claim accuracy. This one measured it, and the first
architecture failed.

### Judgement in a single prompt — rejected

The obvious design asks the model for a verdict directly. Five identical runs
against the clean control model, `temperature=0`:

| run | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|
| verdict | inconclusive | clean | inconclusive | clean | inconclusive |

Three one way, two the other, on byte-identical input. Anything persisted from
this is indefensible: re-run it during a review and the answer changes.

Model choice mattered but did not fix it — same prompt, same data:

| model | false positives on the clean control |
|-------|--------------------------------------|
| `llama-3.3-70b-versatile` | 3 |
| `openai/gpt-oss-120b` | 0 |

Better average quality. Same instability.

### Extraction plus rules — adopted

| model | runs | verdicts |
|-------|------|----------|
| `customer_churn_predictor` | 5 | leakage_detected ×5 |
| `demand_forecast` | 5 | inconclusive ×5 |

Stable on both.

### Why it holds, and where it doesn't

Five extraction fields still wobbled between runs on the churn model — yet the
verdict did not move. The rules are redundant: three of the four observed field
combinations trigger a violation by one path or another. That is a real property,
not luck, and also not a guarantee: a fourth combination exists that did not
occur in five runs and would change the outcome.

The topological layer is immune to this by construction. `reads_target_ancestor`
is computed from the graph, so it returns the same value on every run regardless
of how the model reads a sentence that day.

The more useful finding is *where* the wobble lives:

| feature | description quality | extraction across 5 runs |
|---------|--------------------|--------------------------|
| `promo_response_rate` | window explicitly anchored and closed | identical every run |
| `recency_days` | says "the feature computation date", never anchors it | 3 fields disagreed |

**Extraction variance measures the ambiguity of the text, not noise in the
model.** Where metadata is precise the extractor is deterministic; where it is
vague the extractor splits — which is what a human reviewer does when reading the
same sentence twice.

---

## What it found

Beyond the planted cases, two findings nobody planted.

**A specification error, in this repository.** The clean control originally
described a feature as *"campaigns that ended before the reference date and drew
an order within 14 days of campaign end"*. The agent flagged it. That looked like
a false positive, and the first reaction was to assume the model was wrong.

It wasn't. A campaign ending three days before the reference date has a response
window reaching eleven days past it. The condition closed the *campaign*, not the
*window*. The specification was wrong, written and reviewed by a human who
believed it correct. The fix is in this repository's history.

**Self-referential lineage in the sample data.** `analytics.order_history` lists
itself among its own upstreams. Any recursive walk without a visited set hangs on
the first call. `lineage.py` handles it; worth knowing before writing a traversal
of your own.

---

## Quickstart

Requires a local DataHub quickstart with the `showcase-ecommerce` datapack and a
Groq API key. The free tier is sufficient — the whole project runs at zero cost.

```bash
datahub docker quickstart
datahub datapack load showcase-ecommerce

python -m venv .venv && source .venv/bin/activate
pip install acryl-datahub groq

export GROQ_API_KEY=...

python seed_ml_metadata.py         # adds the ML entities the datapack lacks
python context.py                  # inspect the traversal
python lineage.py --table b2fd91.order_entry_db.analytics.order_history \
                  --column order_status
python temporal.py --model churn   # extract + decide
python agent.py                    # audit and write verdicts back
```

`--runs 5` on `temporal.py` reproduces the variance measurement above.
`--no-write` on `agent.py` analyses without touching the catalog.
Sample outputs live in `examples/` for evaluation without running anything.

### Why a seed script

`showcase-ecommerce` ships datasets, dashboards and column-level lineage, but no
`mlModel`, `mlFeature` or `mlModelDeployment` entities. `seed_ml_metadata.py`
adds a feature store, two models and their deployments, wired into the existing
Snowflake datasets so lineage is continuous from column to deployment.

Nothing in the seeded metadata flags either problem. The poisoned feature's
description is what an honest data engineer would write; the topological one's is
clean by any reading. Both have to be derived.

---

## Decision rules

Readable without running anything, in order of precedence:

| rule | condition | outcome |
|------|-----------|---------|
| `reads-target-without-cutoff` | feature reads the label column, no stated cutoff | violation |
| `forward-from-reference` | window runs forward from the reference date | violation |
| `window-extends-past-scoring` | window stated to reach past the scoring point | violation |
| `temporal-anchoring-undocumented` | direction or anchor not stated | gap |
| `reads-table-carrying-target-column` | sources a table carrying the label's column, at any hop | gap |
| `shares-target-table-no-cutoff-stated` | sources the label's table, no cutoff stated | gap |
| — | nothing fired | clean |

Gaps are not violations. A model whose metadata cannot be audited is a
documentation problem, and reporting it as leakage would be a lie. Keeping the
two apart mattered more than any prompt change: with only one output channel for
"something is off", every uncertainty became an accusation.

---

## What is written back

- a tag on the model: `ml-leakage-risk`, `ml-audit-clean` or `ml-audit-inconclusive`
- an audit trail in `customProperties`: verdict, confidence, finding count, timestamp

Re-running replaces the previous verdict rather than accumulating contradictory
tags.

---

## Limits

- **Only temporal and topological leakage.** Group leakage, duplicate rows across
  splits, and preprocessing applied before splitting are invisible to metadata.
- **Column-level precision is not expressible for features.**
  `MLFeatureProperties.sources` accepts dataset URNs only — the schema declares
  `entityTypes: ['dataset']`. A feature can state which tables it reads, not
  which columns. The topological layer therefore returns *gap*, never
  *violation*: it can show that a feature reads a table carrying the label's
  column, not that it reads that column.
- **Ancestry is capped at six hops.** Deeper chains are truncated silently.
- **Only what the catalog states.** A feature whose implementation contradicts
  its own documentation passes the textual layer. The tool audits declarations.
- **Small evidence base.** Two models, six features, five runs each. Enough to
  reject one architecture in favour of another; not a benchmark.

The planned mitigation for extraction variance is majority voting across three
runs, with any field lacking consensus escalating to a documentation gap — which
turns measured variance into a signal about catalog quality rather than a defect
to hide.

---

## Files

| file | role |
|------|------|
| `context.py` | deterministic graph traversal, context assembly |
| `lineage.py` | column-level ancestry of the label, cycle-safe |
| `temporal.py` | structured extraction + rule engine + variance measurement |
| `agent.py` | end-to-end audit with write-back |
| `seed_ml_metadata.py` | seeds the ML entities the datapack lacks |
