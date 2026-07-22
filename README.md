# leakwatch

Audits deployed ML models for **target leakage** using catalog metadata alone —
no access to training data, no model artifacts, no code.

It reads what a model's features claim to do, checks those claims against the
temporal boundary the label imposes, and walks column-level lineage to catch the
cases where the claims are honest and the problem lies elsewhere.

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
| `fulfilment_exception_rate` | entirely clean | topological only | 90-day window, backward, anchored to the reference date — but sourced from the raw table the label's column comes from |

For the second, the extractor finds nothing wrong, correctly. The finding comes
from the graph:

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
column. A single traversal spans the transformation layer and the warehouse.

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
                                       LLM extracts stated facts, 3× per model,
                                       majority vote per field
                                       ├─ window_length
                                       ├─ direction     backward|forward|unstated
                                       ├─ anchor        reference_date|other|unstated
                                       └─ closes_before_scoring
                                            +
                                       reads_target_column_at_hop  ← from the graph
                                       reads_target_ancestor       ← from the graph
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

Most submissions claim accuracy. This one measured it, and the measurements
rejected two designs before the third held.

All figures below use `openai/gpt-oss-120b` at `temperature=0`, against the same
two seeded models. The architecture is the only variable.

### Baseline: judgement in a single prompt

The obvious design asks the model for a verdict directly. Five identical runs
against the clean control model:

| run | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|
| verdict | inconclusive | inconclusive | inconclusive | clean | clean |

Byte-identical input, two different answers. Anything persisted from this is
indefensible: re-run it during a review and the catalog says something else.

The same architecture on `llama-3.3-70b-versatile` was worse — it reported three
**false positives** on the clean control, flagging features whose descriptions
state retrospective windows. Model choice changes the error rate; it does not
change the instability.

### First correction, still wrong — twice

Moving the decision into code was necessary but not sufficient. Two rules let a
violation fire on the *absence* of evidence:

1. `mentions_target_column` asked the LLM whether a description referred to the
   label's column. That is a judgement dressed as a fact, and it oscillated:
   on one run in five it flipped to yes and turned a documentation gap into an
   accusation.
2. Replacing it with a graph fact fixed that field, but the rule still read
   `closes_before_scoring != "yes"` — which includes `unstated`, the value the
   extractor produces when the text is silent. The clean control was flagged
   `leakage_detected` with high confidence.

Both are the same mistake in two places: **a violation resting on the absence of
counter-evidence rather than on positive evidence.**

Even after fixing that, a single misread sentence was enough. Over five runs,
`promo_response_rate` — whose description explicitly states its window closes
before the reference date — produced three different outcomes:

| run | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|
| outcome | **violation** | clean | gap | gap | clean |

One run in five read "closed before the reference date" as extending past it.

### Adopted: extraction, majority vote, rules

Each model is extracted three times and each field takes the majority value.
Fields without a strict majority collapse to `unstated`, which degrades to a
documentation gap rather than an accusation. Graph-derived fields are not voted
on — they are identical every run by construction.

| model | runs | verdicts |
|-------|------|----------|
| `customer_churn_predictor` | 5 | leakage_detected ×5 |
| `demand_forecast` | 5 | inconclusive ×5 |

Stable on both, with the planted leak found and the control never accused.

### What is still true

Extraction variance has not disappeared; it is absorbed. Individual fields still
disagree between runs where the text is ambiguous, and that disagreement is
itself the signal:

**Where metadata is precise the extractor is deterministic; where it is vague it
splits** — which is what a human reviewer does when reading the same sentence
twice. Voting turns that split into a documentation gap instead of a coin flip.

The topological layer is immune by construction: `reads_target_column_at_hop`
comes from the graph and returns the same value on every run.

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

`--runs 5` on `temporal.py` measures per-field extraction variance.
`--votes N` sets the vote count (default 3; `1` disables voting).
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
| `reads-target-column-past-scoring` | sources the label's own table **and** states a window reaching past scoring | violation |
| `forward-from-reference` | window runs forward from the reference date | violation |
| `window-extends-past-scoring` | window stated to reach past the scoring point | violation |
| `reads-target-table-no-cutoff-stated` | sources the label's own table, cutoff not stated | gap |
| `temporal-anchoring-undocumented` | direction or anchor not stated | gap |
| `reads-table-carrying-target-column` | sources a table carrying the label's column upstream | gap |
| `shares-target-table-no-cutoff-stated` | shares the label's table, no cutoff stated | gap |
| — | nothing fired | clean |

**Every violation requires the metadata to state the problem.** None fires on
`unstated`. This is not a style preference: two earlier versions of these rules
allowed a violation on missing evidence, and both produced false accusations
under extraction variance.

The extraction prompt applies the same rule to itself. `closes_before_scoring`
returns `yes` only when the text states where the window ends — a word like
"trailing" describes direction, not an endpoint — and `no` only when the text
states or explicitly denies a cutoff. Everything else is `unstated`.

Gaps are not violations. A model whose metadata cannot be audited has a
documentation problem, and reporting that as leakage would be a lie.

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
  which columns. Reading the label's *table* is what the rules can establish,
  never reading the label's column itself. This is why the topological layer
  produces gaps, never violations.
- **Three API calls per model per audit.** Voting trades cost for stability. A
  fifty-model catalog is 150 calls per run.
- **Ancestry is capped at six hops.** Deeper chains are truncated silently.
- **Only what the catalog states.** A feature whose implementation contradicts
  its own documentation passes the textual layer. The tool audits declarations.
- **Small evidence base.** Two models, six features, five runs per architecture.
  Enough to reject two designs in favour of a third; not a benchmark.

---

## Files

| file | role |
|------|------|
| `context.py` | deterministic graph traversal, context assembly |
| `lineage.py` | column-level ancestry of the label, cycle-safe |
| `temporal.py` | extraction, majority voting, rule engine, variance measurement |
| `agent.py` | end-to-end audit with write-back |
| `seed_ml_metadata.py` | seeds the ML entities the datapack lacks |
