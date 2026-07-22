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
from temporal import extract_voted, decide, verdict_for, Decision, DEFAULT_LLM

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

# The judgement prompt that used to live here was removed. It produced verdicts
# directly and oscillated 3/2 across identical runs; temporal.py replaced it with
# extraction plus rules. Kept out of the file rather than commented out, so there
# is one way to reach a verdict, not two.


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



def audit(ctx: ModelContext, api_key: str, llm: str = DEFAULT_LLM,
          graph=None, votes: int = 3) -> AuditResult:
    """Extract by majority vote, then apply rules.

    The verdict never comes from the model, and no single misread sentence can
    produce one: a field needs a strict majority across runs to count.
    """
    try:
        specs = extract_voted(ctx, api_key, llm, graph, votes)
    except ImportError:
        return AuditResult(ctx.urn, ctx.name, "inconclusive", "low", "",
                           error="groq package not installed: pip install groq")
    except Exception as e:  # noqa: BLE001
        return AuditResult(ctx.urn, ctx.name, "inconclusive", "low", "",
                           error=f"{type(e).__name__}: {str(e)[:150]}")

    decisions = [decide(s) for s in specs]
    violations = [d for d in decisions if d.outcome == "violation"]
    gaps = [d for d in decisions if d.outcome == "gap"]

    findings = [
        {"feature": d.feature, "severity": "critical", "rule": d.rule,
         "evidence": d.detail, "reasoning": d.detail}
        for d in violations
    ]
    doc_gaps = [
        {"subject": d.feature, "gap": d.rule, "why_it_matters": d.detail}
        for d in gaps
    ]

    verdict = verdict_for(decisions)
    if verdict == "leakage_detected":
        summary = (f"{len(violations)} feature(s) cross the prediction boundary "
                   f"stated by the label.")
        rec = ("Restrict the flagged features to data available before the "
               "scoring date, then retrain.")
    elif verdict == "inconclusive":
        summary = (f"No violation proven, but {len(gaps)} feature(s) lack the "
                   f"metadata needed to verify temporal safety.")
        rec = ("Document the anchoring and cutoff of the listed features so the "
               "audit can reach a verdict.")
    else:
        summary = "All features state temporal boundaries consistent with the label."
        rec = "No action required."

    # Confidence reflects how much of the verdict rests on the graph rather than
    # on reading prose: topological findings are identical across runs.
    from_graph = any(s.reads_target_ancestor for s in specs)
    confidence = "high" if (violations or from_graph) else "medium"

    return AuditResult(
        model_urn=ctx.urn, model_name=ctx.name, verdict=verdict,
        confidence=confidence, summary=summary, findings=findings,
        documentation_gaps=doc_gaps, recommendation=rec,
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
        out.append(f"      rule: {f.get('rule')}")
        out.append(f"      {f.get('evidence')}")
    if r.documentation_gaps:
        out.append(f"\n  documentation gaps ({len(r.documentation_gaps)})")
        for g in r.documentation_gaps:
            out.append(f"      {g.get('subject')}  [{g.get('gap')}]")
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
    ap.add_argument("--votes", type=int, default=3,
                    help="extractions to run per model; majority wins")
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
        r = audit(ctx, api_key, args.llm, ex.graph, args.votes)
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
