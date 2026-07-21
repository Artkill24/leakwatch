#!/usr/bin/env python3
"""
Separate reading from deciding.

The judgement-in-one-prompt approach oscillated: five identical runs at
temperature 0 produced three verdicts one way and two the other, and it
oscillated precisely on the ambiguous cases where judgement matters.

The hypothesis here is that extraction is more stable than judgement -- that
asking "what does this text state" varies less than asking "is this a problem".
That is a hypothesis, not a fact, so this module measures it before relying
on it.

    LLM   -> reads descriptions, reports stated facts, never evaluates
    Python-> compares URNs, applies rules, produces the verdict

Usage:
    python temporal.py --model churn                # extract + decide once
    python temporal.py --model demand --runs 5      # measure extraction variance
    python temporal.py --model churn --runs 5 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from context import MLContextExtractor, ModelContext
from lineage import tables_carrying_target

DEFAULT_LLM = "openai/gpt-oss-120b"

# The fields the extractor is asked to fill. Kept small on purpose: every extra
# field is another thing that can wobble between runs.
EXTRACTION_FIELDS = [
    "window_length",
    "direction",
    "anchor",
    "closes_before_scoring",
    "mentions_target_column",
]

EXTRACT_PROMPT = """\
You extract stated facts from feature descriptions. You do not evaluate, judge, \
or decide whether anything is a problem. That is someone else's job.

For each feature you are given a description and the name of the column the \
model's label is computed from. Report only what the description explicitly \
states.

Fields:

window_length
    The aggregation window if the text states one, as a compact string such as \
"90d" or "14d". null if the text states no window.

direction
    "backward" if the window covers events preceding its anchor point.
    "forward" if it covers events following its anchor point.
    "unstated" if the text does not make the direction determinable.

anchor
    "reference_date" if the window is anchored to the prediction, scoring or \
reference date.
    "other" if anchored to some different event (a campaign end, an order date, \
a snapshot).
    "unstated" if the text does not say what the window is anchored to.

closes_before_scoring
    "yes" only if the text states the observation window closes before the \
scoring point.
    "no" if the text states or plainly entails that it extends to or past that \
point.
    "unstated" otherwise.

mentions_target_column
    "yes" if the description refers to the label column by name or by an \
unmistakable description of it.
    "no" otherwise.

Rules:

- Report "unstated" freely. Absence of information is itself information, and \
guessing defeats the purpose of this step.
- Read the description. Do not infer from the feature's name.
- Do not reason about whether a window is safe or unsafe. Only report what it says.

Return only JSON, no prose, no markdown fences:

{
  "features": [
    {
      "feature": "name",
      "window_length": "90d" | null,
      "direction": "backward" | "forward" | "unstated",
      "anchor": "reference_date" | "other" | "unstated",
      "closes_before_scoring": "yes" | "no" | "unstated",
      "mentions_target_column": "yes" | "no"
    }
  ]
}\
"""


@dataclass
class TemporalSpec:
    feature: str
    window_length: Optional[str] = None
    direction: str = "unstated"
    anchor: str = "unstated"
    closes_before_scoring: str = "unstated"
    mentions_target_column: str = "no"
    # filled in by Python, not by the model
    shares_target_table: bool = False
    reads_target_ancestor: Optional[str] = None  # "table (as column, N hop)"

    def key(self) -> tuple:
        """The fields whose stability we care about."""
        return tuple(getattr(self, f) for f in EXTRACTION_FIELDS)


@dataclass
class Decision:
    feature: str
    outcome: str          # violation | gap | clean
    rule: str             # which rule fired
    detail: str = ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract(ctx: ModelContext, api_key: str, llm: str = DEFAULT_LLM,
            graph=None) -> List[TemporalSpec]:
    from groq import Groq

    label_column = ctx.custom_properties.get("label_source_column", "(not documented)")
    label_table = ctx.custom_properties.get("label_source_table", "")

    payload = {
        "label_column": label_column,
        "label_logic": ctx.custom_properties.get("label_logic", "(not documented)"),
        "features": [
            {"name": f.name, "description": f.description or "(no description)"}
            for f in ctx.features
        ],
    }

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=llm,
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)

    # Column ancestry of the label, resolved in the graph. A feature sourcing a
    # table that carries the target column under any name is reading the label,
    # whatever its description says.
    ancestor_tables = {}
    if label_table and label_column != "(not documented)":
        label_urn = next(
            (s.urn for f in ctx.features for s in f.sources if s.name == label_table),
            None,
        )
        if label_urn and graph is not None:
            try:
                ancestor_tables = tables_carrying_target(graph, label_urn, label_column)
            except Exception:  # noqa: BLE001
                ancestor_tables = {}

    # Python decides table overlap; the model never sees URNs.
    sources_by_feature = {
        f.name: {s.name for s in f.sources} for f in ctx.features
    }

    specs = []
    for row in data.get("features", []):
        name = row.get("feature", "")
        spec = TemporalSpec(
            feature=name,
            window_length=row.get("window_length"),
            direction=row.get("direction", "unstated"),
            anchor=row.get("anchor", "unstated"),
            closes_before_scoring=row.get("closes_before_scoring", "unstated"),
            mentions_target_column=row.get("mentions_target_column", "no"),
        )
        if label_table:
            spec.shares_target_table = label_table in sources_by_feature.get(name, set())
        if ancestor_tables:
            for src_short in sources_by_feature.get(name, set()):
                for anc_urn, (anc_col, hops) in ancestor_tables.items():
                    if src_short and src_short in anc_urn:
                        spec.reads_target_ancestor = f"{src_short} (as {anc_col}, {hops} hop)"
                        break
        specs.append(spec)
    return specs


# ---------------------------------------------------------------------------
# Decision -- no model involved past this line
# ---------------------------------------------------------------------------


def decide(spec: TemporalSpec) -> Decision:
    """Apply the rules in order of severity. First match wins."""

    if spec.mentions_target_column == "yes" and spec.closes_before_scoring != "yes":
        return Decision(
            spec.feature, "violation", "reads-target-without-cutoff",
            "feature reads the label column and does not state a cutoff before scoring",
        )

    if spec.direction == "forward" and spec.anchor == "reference_date":
        return Decision(
            spec.feature, "violation", "forward-from-reference",
            "window runs forward from the reference date",
        )

    if spec.closes_before_scoring == "no":
        return Decision(
            spec.feature, "violation", "window-extends-past-scoring",
            "window is stated to extend to or past the scoring point",
        )

    if spec.direction == "unstated" or spec.anchor == "unstated":
        missing = [f for f in ("direction", "anchor")
                   if getattr(spec, f) == "unstated"]
        return Decision(
            spec.feature, "gap", "temporal-anchoring-undocumented",
            f"description does not state: {', '.join(missing)}",
        )

    if spec.reads_target_ancestor:
        return Decision(
            spec.feature, "gap", "reads-table-carrying-target-column",
            f"sources {spec.reads_target_ancestor}; may read the label under "
            f"another address regardless of what the description states",
        )

    if spec.shares_target_table and spec.closes_before_scoring == "unstated":
        return Decision(
            spec.feature, "gap", "shares-target-table-no-cutoff-stated",
            "sources the label's table without stating where the window closes",
        )

    return Decision(spec.feature, "clean", "no-rule-fired")


def verdict_for(decisions: List[Decision]) -> str:
    if any(d.outcome == "violation" for d in decisions):
        return "leakage_detected"
    if any(d.outcome == "gap" for d in decisions):
        return "inconclusive"
    return "clean"


# ---------------------------------------------------------------------------
# Variance measurement
# ---------------------------------------------------------------------------


def measure(ctx: ModelContext, api_key: str, llm: str, runs: int, graph=None) -> dict:
    """Run extraction N times and report where it disagrees with itself."""
    all_specs: List[List[TemporalSpec]] = []
    verdicts: List[str] = []

    for i in range(runs):
        try:
            specs = extract(ctx, api_key, llm, graph)
        except Exception as e:  # noqa: BLE001
            print(f"  run {i+1}: FAILED ({type(e).__name__})", file=sys.stderr)
            continue
        all_specs.append(specs)
        verdicts.append(verdict_for([decide(s) for s in specs]))
        print(f"  run {i+1}: {verdicts[-1]}", file=sys.stderr)

    if not all_specs:
        return {"error": "all runs failed"}

    # per-feature, per-field agreement
    fields_report = {}
    feature_names = [s.feature for s in all_specs[0]]
    for name in feature_names:
        per_field = {}
        for fld in EXTRACTION_FIELDS:
            values = []
            for run in all_specs:
                match = next((s for s in run if s.feature == name), None)
                if match:
                    values.append(str(getattr(match, fld)))
            counts = Counter(values)
            per_field[fld] = {
                "stable": len(counts) == 1,
                "values": dict(counts),
            }
        fields_report[name] = per_field

    unstable = [
        f"{feat}.{fld}"
        for feat, flds in fields_report.items()
        for fld, r in flds.items()
        if not r["stable"]
    ]

    return {
        "runs": len(all_specs),
        "verdicts": dict(Counter(verdicts)),
        "verdict_stable": len(set(verdicts)) == 1,
        "unstable_fields": unstable,
        "field_detail": fields_report,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gms", default="http://localhost:8080")
    ap.add_argument("--model", default=None)
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--runs", type=int, default=1,
                    help="run extraction N times and report variance")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is not set.", file=sys.stderr)
        return 1

    try:
        ex = MLContextExtractor(args.gms)
        ex.graph.test_connection()
    except Exception as e:  # noqa: BLE001
        print(f"Cannot reach DataHub at {args.gms}: {type(e).__name__}", file=sys.stderr)
        return 1

    urns = [u for u in ex.list_models()
            if not args.model or args.model.lower() in u.lower()]
    if not urns:
        print("No matching models.", file=sys.stderr)
        return 1

    for urn in urns:
        ctx = ex.extract(urn)
        if not ctx:
            continue

        if args.runs > 1:
            print(f"\n{ctx.name}: measuring {args.runs} runs", file=sys.stderr)
            report = measure(ctx, api_key, args.llm, args.runs, ex.graph)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(f"\nMODEL {ctx.name}")
                print(f"  verdicts:  {report['verdicts']}")
                print(f"  stable:    {report['verdict_stable']}")
                if report["unstable_fields"]:
                    print(f"  wobbled:   {len(report['unstable_fields'])} field(s)")
                    for u in report["unstable_fields"]:
                        feat, fld = u.split(".", 1)
                        vals = report["field_detail"][feat][fld]["values"]
                        print(f"      {u}: {vals}")
                else:
                    print("  wobbled:   nothing -- extraction was identical every run")
            continue

        specs = extract(ctx, api_key, args.llm, ex.graph)
        decisions = [decide(s) for s in specs]
        print(f"\nMODEL {ctx.name}")
        print(f"  verdict: {verdict_for(decisions)}")
        for s, d in zip(specs, decisions):
            mark = {"violation": "[!!]", "gap": "[? ]", "clean": "[ok]"}[d.outcome]
            print(f"\n  {mark} {s.feature}")
            print(f"      window={s.window_length} dir={s.direction} "
                  f"anchor={s.anchor} closes={s.closes_before_scoring} "
                  f"target_col={s.mentions_target_column} "
                  f"same_table={s.shares_target_table}")
            if s.reads_target_ancestor:
                print(f"      ancestor: {s.reads_target_ancestor}")
            print(f"      rule: {d.rule}")
            if d.detail:
                print(f"      {d.detail}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
