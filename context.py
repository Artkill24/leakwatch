#!/usr/bin/env python3
"""
Extract the full context surrounding an ML model from DataHub's graph.

Design note
-----------
The traversal here is deterministic: no LLM is involved in walking the graph.
We resolve model -> features -> source datasets -> upstream datasets by reading
typed aspects directly, assemble one compact context bundle, and only then hand
that bundle to a model for reasoning.

The alternative -- giving an LLM a fistful of MCP tools and letting it wander
the graph -- costs a round trip per hop, is non-deterministic, and is hard to
test. Walking the graph is a solved problem; judgement is the part worth
spending tokens on.

Usage:
    python context.py                      # human-readable, all models
    python context.py --json               # machine-readable bundle
    python context.py --model customer_churn_predictor
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from datahub.ingestion.graph.client import DataHubGraph, DatahubClientConfig
from datahub.metadata.urns import DatasetUrn, MlFeatureUrn, MlModelUrn
import datahub.metadata.schema_classes as sc

# Cap on schema fields carried per dataset. Column names are what let a model
# recognise that a table holds an outcome, but showcase-ecommerce alone has 873
# fields and we are not going to pay for all of them.
MAX_FIELDS_PER_DATASET = 25


# ---------------------------------------------------------------------------
# Context nodes
# ---------------------------------------------------------------------------


@dataclass
class DatasetNode:
    urn: str
    name: str
    platform: str
    description: Optional[str] = None
    field_names: List[str] = field(default_factory=list)
    field_count: int = 0
    upstreams: List[str] = field(default_factory=list)


@dataclass
class FeatureNode:
    urn: str
    name: str
    description: Optional[str] = None
    data_type: Optional[str] = None
    sources: List[DatasetNode] = field(default_factory=list)


@dataclass
class DeploymentNode:
    urn: str
    name: str
    status: Optional[str] = None
    version: Optional[str] = None


@dataclass
class ModelContext:
    urn: str
    name: str
    description: Optional[str] = None
    model_type: Optional[str] = None
    version: Optional[str] = None
    custom_properties: Dict[str, str] = field(default_factory=dict)
    training_metrics: Dict[str, str] = field(default_factory=dict)
    hyper_params: Dict[str, str] = field(default_factory=dict)
    features: List[FeatureNode] = field(default_factory=list)
    deployments: List[DeploymentNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class MLContextExtractor:
    def __init__(self, gms: str = "http://localhost:8080", token: Optional[str] = None):
        self.graph = DataHubGraph(DatahubClientConfig(server=gms, token=token))
        self._dataset_cache: Dict[str, DatasetNode] = {}

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _short_platform(platform_urn: str) -> str:
        return platform_urn.rsplit(":", 1)[-1] if platform_urn else "unknown"

    def _dataset(self, urn: str) -> DatasetNode:
        """Resolve a dataset URN into a node, with schema and upstreams."""
        if urn in self._dataset_cache:
            return self._dataset_cache[urn]

        try:
            parsed = DatasetUrn.from_string(urn)
            name = parsed.name
            platform = self._short_platform(str(parsed.platform))
        except Exception:  # noqa: BLE001 - malformed urn shouldn't kill the run
            name, platform = urn, "unknown"

        node = DatasetNode(urn=urn, name=name, platform=platform)

        props = self.graph.get_aspect(urn, sc.DatasetPropertiesClass)
        if props:
            node.description = props.description

        schema = self.graph.get_aspect(urn, sc.SchemaMetadataClass)
        if schema and schema.fields:
            node.field_count = len(schema.fields)
            node.field_names = [f.fieldPath for f in schema.fields[:MAX_FIELDS_PER_DATASET]]

        lineage = self.graph.get_aspect(urn, sc.UpstreamLineageClass)
        if lineage and lineage.upstreams:
            node.upstreams = [u.dataset for u in lineage.upstreams]

        self._dataset_cache[urn] = node
        return node

    def _feature(self, urn: str) -> FeatureNode:
        try:
            name = MlFeatureUrn.from_string(urn).name
        except Exception:  # noqa: BLE001
            name = urn

        node = FeatureNode(urn=urn, name=name)

        props = self.graph.get_aspect(urn, sc.MLFeaturePropertiesClass)
        if props:
            node.description = props.description
            node.data_type = props.dataType
            for src in props.sources or []:
                node.sources.append(self._dataset(src))

        return node

    def _deployment(self, urn: str) -> DeploymentNode:
        name = urn.split(",")[-2] if "," in urn else urn
        node = DeploymentNode(urn=urn, name=name)
        props = self.graph.get_aspect(urn, sc.MLModelDeploymentPropertiesClass)
        if props:
            node.status = props.status
            node.version = props.version.versionTag if props.version else None
        return node

    # -- public -----------------------------------------------------------

    def list_models(self, limit: int = 100) -> List[str]:
        return self.graph.list_all_entity_urns("mlModel", 0, limit) or []

    def extract(self, model_urn: str) -> Optional[ModelContext]:
        props = self.graph.get_aspect(model_urn, sc.MLModelPropertiesClass)
        if props is None:
            return None

        try:
            name = props.name or MlModelUrn.from_string(model_urn).name
        except Exception:  # noqa: BLE001
            name = model_urn

        ctx = ModelContext(
            urn=model_urn,
            name=name,
            description=props.description,
            model_type=props.type,
            version=props.version.versionTag if props.version else None,
            custom_properties=dict(props.customProperties or {}),
            training_metrics={m.name: m.value for m in (props.trainingMetrics or [])},
            hyper_params={h.name: h.value for h in (props.hyperParams or [])},
        )

        for furn in props.mlFeatures or []:
            ctx.features.append(self._feature(furn))

        for durn in props.deployments or []:
            ctx.deployments.append(self._deployment(durn))

        return ctx


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(ctx: ModelContext) -> str:
    out = []
    out.append(f"MODEL  {ctx.name}  (v{ctx.version or '?'}, {ctx.model_type or '?'})")
    if ctx.description:
        out.append(f"       {ctx.description}")

    if ctx.custom_properties:
        out.append("\n  properties")
        for k, v in ctx.custom_properties.items():
            out.append(f"    {k}: {v}")

    if ctx.training_metrics:
        metrics = "  ".join(f"{k}={v}" for k, v in ctx.training_metrics.items())
        out.append(f"\n  training metrics: {metrics}")

    for d in ctx.deployments:
        out.append(f"  deployed: {d.name} [{d.status}]")

    out.append(f"\n  features ({len(ctx.features)})")
    for f in ctx.features:
        out.append(f"    - {f.name} ({f.data_type})")
        if f.description:
            out.append(f"        {f.description}")
        for s in f.sources:
            up = f"  <- {len(s.upstreams)} upstream" if s.upstreams else ""
            out.append(f"        from {s.platform}:{s.name} ({s.field_count} cols){up}")
            if s.field_names:
                preview = ", ".join(s.field_names[:8])
                more = "..." if s.field_count > 8 else ""
                out.append(f"          cols: {preview}{more}")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gms", default="http://localhost:8080")
    ap.add_argument("--token", default=None)
    ap.add_argument("--model", default=None, help="filter by model name substring")
    ap.add_argument("--json", action="store_true", help="emit the raw context bundle")
    args = ap.parse_args()

    # Two distinct failures, two distinct messages: a server that isn't there
    # is not the same as a server with nothing in it.
    try:
        ex = MLContextExtractor(args.gms, args.token)
        ex.graph.test_connection()
    except Exception as e:  # noqa: BLE001
        print(f"Cannot reach DataHub at {args.gms}", file=sys.stderr)
        print(f"  {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        print("\nStart it with: datahub docker quickstart --no-pull-images",
              file=sys.stderr)
        return 1

    urns = ex.list_models()
    if not urns:
        print("Connected, but no mlModel entities exist.", file=sys.stderr)
        print("Seed them with: python seed_ml_metadata.py", file=sys.stderr)
        return 1

    contexts = []
    for urn in urns:
        if args.model and args.model.lower() not in urn.lower():
            continue
        ctx = ex.extract(urn)
        if ctx:
            contexts.append(ctx)

    if args.json:
        print(json.dumps([c.to_dict() for c in contexts], indent=2))
    else:
        for c in contexts:
            print(render(c))
            print("\n" + "-" * 70 + "\n")
        print(f"{len(contexts)} model(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
