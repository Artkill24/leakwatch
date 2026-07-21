# leakwatch

Audits deployed ML models for **target leakage** using catalog metadata alone —
no access to training data, no model artifacts, no code.

It walks DataHub's lineage from a model to its features to the columns those
features read, then checks whether any feature crosses the temporal boundary
its own label imposes.

---

## The problem

Target leakage is expensive and quiet. A feature encodes something that will
only be knowable *after* the prediction point, the model scores beautifully in
validation, and it degrades the moment it meets production. By the time anyone
notices, it has been making decisions for months.

Catalogs already hold what is needed to catch this: what the label is computed
from, what each feature reads, and how the windows are anchored. Nobody reads it,
because reading it across dozens of models is tedious and nobody has the time.

That is a job for an agent.

---

## How it works

```
DataHub graph                     deterministic traversal (no LLM)
  mlModel
    └─ mlFeature ────────────────► context.py
         └─ dataset                  resolves model → features → datasets
              └─ schemaField         → columns → upstream lineage
                                            │
                                            ▼
                                     temporal.py
                                       LLM extracts stated facts only
                                       ├─ window_length
                                       ├─ direction        backward|forward|unstated
                                       ├─ anchor           reference_date|other|unstated
                                       ├─ closes_before_scoring
                                       └─ mentions_target_column
                                            │
                                            ▼
                                     rules in Python
                                       no model past this line
                                            │
                                            ▼
                                     write-back to DataHub
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
combinations trigger a violation by one path or another.

That is a real property, not luck. It is also not a guarantee: a fourth
combination exists that did not occur in five runs and would change the outcome.

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

The audit was built against a seeded scenario with one deliberately poisoned
feature and one clean control. It found the planted case, with evidence quoting
the metadata rather than the suspiciously high AUC.

It also found something nobody planted.

The clean control originally described a feature as *"campaigns that ended
before the reference date and drew an order within 14 days of campaign end"*.
The agent flagged it. That looked like a false positive, and the first reaction
was to assume the model was wrong.

It wasn't. A campaign ending three days before the reference date has a response
window reaching eleven days past it. The condition closed the *campaign*, not the
*window*. The specification was wrong, written and reviewed by a human who
believed it was correct.

The fix — closing the window rather than the campaign — is in the seed script's
history.

---

## Quickstart

Requires a local DataHub quickstart with the `showcase-ecommerce` datapack, and
a Groq API key (free tier is sufficient — the whole project runs at zero cost).

```bash
datahub docker quickstart
datahub datapack load showcase-ecommerce

python -m venv .venv && source .venv/bin/activate
pip install acryl-datahub groq

export GROQ_API_KEY=...

python seed_ml_metadata.py       # adds ML entities the datapack lacks
python context.py                # inspect the traversal
python temporal.py --model churn # extract + decide
python agent.py                  # audit and write verdicts back
```

`--runs 5` on `temporal.py` reproduces the variance measurement above.
`--no-write` on `agent.py` analyses without touching the catalog.

### Why a seed script

`showcase-ecommerce` ships datasets, dashboards and column-level lineage, but no
`mlModel`, `mlFeature` or `mlModelDeployment` entities. `seed_ml_metadata.py`
adds a feature store, two models and their deployments, wired into the existing
Snowflake datasets so lineage is continuous from column to deployment.

Nothing in the seeded metadata flags the leak. The poisoned feature's
description is what an honest data engineer would write. The agent has to derive
the problem from the lineage, not read it off a field.

---

## Decision rules

Readable without running anything, in order of precedence:

| rule | condition | outcome |
|------|-----------|---------|
| `reads-target-without-cutoff` | feature reads the label column, no stated cutoff | violation |
| `forward-from-reference` | window runs forward from the reference date | violation |
| `window-extends-past-scoring` | window stated to reach past the scoring point | violation |
| `temporal-anchoring-undocumented` | direction or anchor not stated | gap |
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

- **Only temporal leakage.** Group leakage, duplicate rows across splits and
  preprocessing applied before splitting are invisible to metadata.
- **Only what the catalog states.** A feature whose implementation contradicts
  its documentation will pass. The tool audits the declaration, not the code.
- **Small evidence base.** Two models, five features, five runs each. Enough to
  reject one architecture in favour of another; not a benchmark.
- **Extraction is not fully deterministic.** Redundant rules absorb the observed
  variance, but a combination exists that would not be absorbed.

The planned mitigation is majority voting across three extractions, with any
field lacking consensus escalating to a documentation gap — which turns the
measured variance into a signal about catalog quality instead of a defect to
hide.

---

## Files

| file | role |
|------|------|
| `context.py` | deterministic graph traversal, context assembly |
| `temporal.py` | structured extraction + rule engine + variance measurement |
| `agent.py` | end-to-end audit with write-back |
| `seed_ml_metadata.py` | seeds the ML entities the datapack lacks |
