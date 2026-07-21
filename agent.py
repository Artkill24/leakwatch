#!/usr/bin/env python3
"""
Audit deployed ML models for training-data leakage, using metadata alone.

How it works
------------
1. context.py walks the graph deterministically and assembles a context bundle
   (model, features, source datasets, columns, label definition).
2. One LLM call reasons over that bundle and returns a structured verdict.
3. The verdict is written back into DataHub as tags plus an audit trail on the
   model, so the finding lives in the catalog rather than in a terminal.

The prompt deliberately contains no hint about this particular dataset. It
describes what leakage *is* and asks the model to reason about temporal
boundaries. A prompt that said "look for features sourced from the label table"
would find the planted case and nothing else.

Usage:
    export GROQ_API_KEY=...
    python agent.py                      # audit every deployed model
    python agent.py --model churn        # filter
    python agent.py --no-write           # analyse, don't touch the catalog
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.urns import TagUrn
import datahub.metadata.schema_classes as sc

from context import MLContextExtractor, ModelContext

DEFAULT_LLM = "openai/gpt-oss-120b"

TAGS = {
    "ml-leakage-risk": (
        "Target leakage suspected: a feature appears to encode information "
        "unavailable at prediction time.",
        "#C4302B",
    ),
    "ml-audit-clean": (
        "Automated leakage audit found no temporal boundary violations.",
        "#2E7D32",
    ),
    "ml-audit-inconclusive": (
        "Automated leakage audit could not reach a verdict; metadata is "
        "insufficient.",
        "#F9A825",
    ),
}

SYSTEM_PROMPT = """\
You are auditing a deployed machine learning model for training-data leakage, \
working only from catalog metadata. You cannot see the data, only what the \
catalog states about it.

Target leakage occurs when a feature encodes information that would not be \
available at the moment a prediction is made.

EVIDENCE RULES. These govern every finding:

1. A finding must quote the specific metadata text that establishes the \
violation. If you cannot quote it, you do not have a finding.

2. Sharing a source table with the label is not evidence. Tables routinely hold \
both precursors and outcomes. What matters is whether a stated computation \
crosses the prediction boundary.

3. When a feature's description states a temporal restriction -- a trailing \
window, a cutoff, a filter on events preceding the reference date -- treat that \
statement as binding. Do not hypothesise that the implementation might violate \
its own documentation. You are auditing what the catalog declares.

4. Speculation is not a finding. If your reasoning needs "may", "could" or \
"might" to stand up, and no quoted metadata supports it, leave it out.

5. Implausibly strong training metrics corroborate a finding that already rests \
on other evidence. Alone they prove nothing; some problems are genuinely easy.

What leakage actually looks like:
- a feature derived from an event occurring after the prediction point
- a feature reading the field the label is defined on, without the temporal \
restriction the label imposes
- a feature whose stated aggregation window extends past the scoring date
- a feature that restates the outcome rather than preceding it

Where metadata is ambiguous or absent rather than wrong, that is a \
documentation gap, not a leak. Report it under documentation_gaps. A catalog \
that does not state its temporal boundaries cannot be audited, and saying so is \
worth more than a guess.

Set verdict to leakage_detected only if findings is non-empty. If findings is \
empty but documentation_gaps is not, the verdict is inconclusive. If both are \
empty, the verdict is clean.

Return only JSON, no prose, no markdown fences, in exactly this shape:

{
  "verdict": "leakage_detected" | "clean" | "inconclusive",
  "confidence": "high" | "medium" | "low",
  "summary": "one or two sentences",
  "findings": [
    {
      "feature": "feature name",
      "severity": "critical" | "warning",
      "evidence": "quoted metadata text establishing the violation",
      "reasoning": "why that text implies a boundary crossing"
    }
  ],
  "documentation_gaps": [
    {
      "subject": "feature or model name",
      "gap": "what the metadata fails to state",
      "why_it_matters": "what could not be verified as a result"
    }
  ],
  "recommendation": "what the owning team should do next"
}\
"""


@dataclass
class AuditResult:
    model_urn: str
    model_name: str
    verdict: str
    confidence: str
    summary: str
    findings: List[dict] = field(default_factory=list)
    documentation_gaps: List[dict] = field(default_factory=list)
    recommendation: str = ""
    error: Optional[str] = None

    @property
    def tag(self) -> str:
        return {
            "leakage_detected": "ml-leakage-risk",
            "clean": "ml-audit-clean",
        }.get(self.verdict, "ml-audit-inconclusive")


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


def build_evidence(ctx: ModelContext) -> dict:
    """Reduce the context bundle to what the audit actually needs.

    Dropping hyperparameters, URNs and deployment plumbing keeps the prompt
    small and stops the model from anchoring on irrelevant detail.
    """
    return {
        "model": {
            "name": ctx.name,
            "description": ctx.description,
            "type": ctx.model_type,
            "label": {
                k: v
                for k, v in ctx.custom_properties.items()
                if k.startswith("label")
            },
            "training_window": ctx.custom_properties.get("training_window"),
            "training_metrics": ctx.training_metrics,
        },
        "features": [
            {
                "name": f.name,
                "description": f.description,
                "data_type": f.data_type,
                "sources": [
                    {
                        "table": s.name,
                        "platform": s.platform,
                        "columns": s.field_names,
                        "column_count": s.field_count,
                    }
                    for s in f.sources
                ],
            }
            for f in ctx.features
        ],
    }


def audit(ctx: ModelContext, api_key: str, llm: str = DEFAULT_LLM) -> AuditResult:
    try:
        from groq import Groq
    except ImportError:
        return AuditResult(
            ctx.urn, ctx.name, "inconclusive", "low", "",
            error="groq package not installed: pip install groq",
        )

    evidence = build_evidence(ctx)

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=llm,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(evidence, indent=2)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        return AuditResult(ctx.urn, ctx.name, "inconclusive", "low", "",
                           error=f"{type(e).__name__}: {str(e)[:150]}")

    try:
        data = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
    except json.JSONDecodeError:
        return AuditResult(ctx.urn, ctx.name, "inconclusive", "low", "",
                           error=f"model returned non-JSON: {raw[:150]}")

    return AuditResult(
        model_urn=ctx.urn,
        model_name=ctx.name,
        verdict=data.get("verdict", "inconclusive"),
        confidence=data.get("confidence", "low"),
        summary=data.get("summary", ""),
        findings=data.get("findings", []) or [],
        documentation_gaps=data.get("documentation_gaps", []) or [],
        recommendation=data.get("recommendation", ""),
    )


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


def ensure_tags(emitter: DatahubRestEmitter) -> None:
    """Create the tag entities so they render with names and colours."""
    for name, (desc, colour) in TAGS.items():
        emitter.emit(
            MetadataChangeProposalWrapper(
                entityUrn=str(TagUrn(name)),
                aspect=sc.TagPropertiesClass(
                    name=name, description=desc, colorHex=colour
                ),
            )
        )


def write_back(emitter: DatahubRestEmitter, graph, result: AuditResult) -> None:
    """Apply the verdict tag and record an audit trail on the model.

    Existing tags are preserved except for our own, which are replaced so that
    re-running the audit doesn't accumulate contradictory verdicts.
    """
    existing = graph.get_aspect(result.model_urn, sc.GlobalTagsClass)
    kept = [
        t
        for t in (existing.tags if existing else [])
        if t.tag.rsplit(":", 1)[-1] not in TAGS
    ]
    kept.append(sc.TagAssociationClass(tag=str(TagUrn(result.tag))))

    emitter.emit(
        MetadataChangeProposalWrapper(
            entityUrn=result.model_urn,
            aspect=sc.GlobalTagsClass(tags=kept),
        )
    )

    # Audit trail goes into customProperties via read-modify-write, so the
    # seeded properties survive.
    props = graph.get_aspect(result.model_urn, sc.MLModelPropertiesClass)
    if props is None:
        return
    custom = dict(props.customProperties or {})
    custom.update(
        {
            "audit_verdict": result.verdict,
            "audit_confidence": result.confidence,
            "audit_summary": result.summary[:500],
            "audit_findings_count": str(len(result.findings)),
            "audit_gaps_count": str(len(result.documentation_gaps)),
            "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    props.customProperties = custom
    emitter.emit(
        MetadataChangeProposalWrapper(entityUrn=result.model_urn, aspect=props)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SEV_MARK = {"critical": "[!!]", "warning": "[! ]", "info": "[  ]"}


def render(r: AuditResult) -> str:
    if r.error:
        return f"MODEL {r.model_name}\n  ERROR  {r.error}"

    out = [f"MODEL {r.model_name}",
           f"  verdict: {r.verdict}  (confidence: {r.confidence})",
           f"  {r.summary}"]
    for f in r.findings:
        out.append(f"\n  {SEV_MARK.get(f.get('severity'), '[  ]')} {f.get('feature')}")
        out.append(f"      evidence:  {f.get('evidence')}")
        out.append(f"      reasoning: {f.get('reasoning')}")
    if r.documentation_gaps:
        out.append(f"\n  documentation gaps ({len(r.documentation_gaps)})")
        for g in r.documentation_gaps:
            out.append(f"      {g.get('subject')}: {g.get('gap')}")
            out.append(f"        -> {g.get('why_it_matters')}")
    if r.recommendation:
        out.append(f"\n  recommendation: {r.recommendation}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gms", default="http://localhost:8080")
    ap.add_argument("--token", default=None)
    ap.add_argument("--model", default=None, help="filter by name substring")
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--no-write", action="store_true",
                    help="analyse without modifying the catalog")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is not set. Get a free key at console.groq.com",
              file=sys.stderr)
        return 1

    try:
        ex = MLContextExtractor(args.gms, args.token)
        ex.graph.test_connection()
    except Exception as e:  # noqa: BLE001
        print(f"Cannot reach DataHub at {args.gms}", file=sys.stderr)
        print(f"  {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return 1

    urns = [u for u in ex.list_models()
            if not args.model or args.model.lower() in u.lower()]
    if not urns:
        print("No matching mlModel entities.", file=sys.stderr)
        return 1

    emitter = None
    if not args.no_write:
        emitter = DatahubRestEmitter(gms_server=args.gms, token=args.token)
        ensure_tags(emitter)

    results = []
    for urn in urns:
        ctx = ex.extract(urn)
        if not ctx:
            continue
        r = audit(ctx, api_key, args.llm)
        results.append(r)
        if emitter and not r.error:
            write_back(emitter, ex.graph, r)

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        for r in results:
            print(render(r))
            print("\n" + "-" * 70 + "\n")
        flagged = sum(1 for r in results if r.verdict == "leakage_detected")
        print(f"{len(results)} model(s) audited, {flagged} flagged.")
        if emitter:
            print(f"Verdicts written to the catalog. See {args.gms.replace('8080','9002')}/mlModels")

    return 0


if __name__ == "__main__":
    sys.exit(main())
