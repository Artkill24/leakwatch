#!/usr/bin/env python3
"""
Seed ML metadata into a local DataHub quickstart.

The showcase-ecommerce datapack ships datasets, dashboards and column-level
lineage, but no ML entities. This script adds a feature store, two models and
their deployments, wired into the existing Snowflake datasets so that the full
lineage chain becomes traversable:

    dataset -> mlFeature -> mlModel -> mlModelDeployment

One of the two models is deliberately built on a leaky feature. Nothing in the
metadata says "this is a leak" -- the leak is only visible by walking the
lineage and reasoning about what the source table actually contains. That is the
point: the agent has to derive it, not read it off a flag.

Usage:
    python3 seed_ml_metadata.py
    python3 seed_ml_metadata.py --gms http://localhost:8080 --dry-run
"""

import argparse
import sys
import time

from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.urns import (
    MlFeatureTableUrn,
    MlFeatureUrn,
    MlModelDeploymentUrn,
    MlModelUrn,
    MlPrimaryKeyUrn,
)
import datahub.metadata.schema_classes as sc

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Prefix used by the showcase-ecommerce datapack. If you loaded a different
# datapack, check an existing URN in the UI and adjust.
PREFIX = "b2fd91"
DB = f"{PREFIX}.order_entry_db"

FEATURE_TABLE = "ecommerce_customer_churn_features"
FEATURE_PLATFORM = "feast"
MODEL_PLATFORM = "mlflow"
ENV = "PROD"


def snowflake(table: str) -> str:
    return f"urn:li:dataset:(urn:li:dataPlatform:snowflake,{DB}.{table},{ENV})"


CUSTOMERS = snowflake("order_entry.customers")
ORDER_ITEMS = snowflake("order_entry.order_items")
ORDER_HISTORY = snowflake("analytics.order_history")
PROMOTIONS = snowflake("order_entry.promotions")
# Raw table upstream of analytics.order_history. Both carry order_status.
ORDER_DETAILS = snowflake("analytics.order_details")

# --------------------------------------------------------------------------
# Features
#
# `sources` is what creates dataset -> feature lineage. The last feature is the
# poisoned one: refunds issued after a cancellation are only known once the
# customer has already churned, so the value cannot exist at prediction time.
# --------------------------------------------------------------------------

FEATURES = [
    {
        "name": "recency_days",
        "type": sc.MLFeatureDataTypeClass.CONTINUOUS,
        "description": "Days elapsed between the customer's most recent completed order and the feature computation date.",
        "sources": [CUSTOMERS, ORDER_HISTORY],
    },
    {
        "name": "order_frequency_90d",
        "type": sc.MLFeatureDataTypeClass.COUNT,
        "description": "Number of distinct orders placed by the customer in the trailing 90-day window.",
        "sources": [ORDER_ITEMS],
    },
    {
        "name": "avg_basket_value",
        "type": sc.MLFeatureDataTypeClass.CONTINUOUS,
        "description": "Mean order value across the customer's completed orders in the trailing 90-day window.",
        "sources": [ORDER_ITEMS],
    },
    {
        "name": "promo_response_rate",
        "type": sc.MLFeatureDataTypeClass.CONTINUOUS,
        "description": (
            "Share of promotional campaigns whose full 14-day response window "
            "closed before the reference date and drew at least one order "
            "inside that window, computed over the trailing 180-day window."
        ),
        "sources": [PROMOTIONS, ORDER_ITEMS],
    },
    {
        "name": "post_cancellation_refund_total",
        "type": sc.MLFeatureDataTypeClass.CONTINUOUS,
        "description": (
            "Sum of order_total across the customer's orders whose order_status "
            "is 'cancelled', computed over the full order_history snapshot "
            "rather than restricted to the pre-scoring window."
        ),
        "sources": [ORDER_HISTORY],
    },
    {
        # Clean on a plain reading: the window is anchored and closed. The
        # problem is only visible in the graph -- order_details is the raw
        # table order_history was built from, and carries order_status, the
        # column the label is defined on.
        "name": "fulfilment_exception_rate",
        "type": sc.MLFeatureDataTypeClass.CONTINUOUS,
        "description": (
            "Share of the customer's orders in the trailing 90-day window "
            "preceding the reference date that required manual intervention "
            "during fulfilment, computed from order detail records."
        ),
        "sources": [ORDER_DETAILS],
    },
]

PRIMARY_KEY = {
    "name": "customer_id",
    "type": sc.MLFeatureDataTypeClass.TEXT,
    "description": "Surrogate key of the customer entity.",
    "sources": [CUSTOMERS],
}

# --------------------------------------------------------------------------
# Models
#
# churn_predictor consumes every feature, including the poisoned one, and posts
# an AUC that is implausibly high for churn. demand_forecast is the negative
# control: clean inputs, believable metrics. An agent that flags both is not
# actually reasoning.
# --------------------------------------------------------------------------

MODELS = [
    {
        "name": "customer_churn_predictor",
        "description": (
            "Gradient-boosted classifier predicting whether a customer will "
            "churn within 30 days of the scoring date."
        ),
        "version": "3.2.0",
        "type": "gradient_boosted_trees",
        "features": [f["name"] for f in FEATURES],
        "metrics": [
            ("auc_roc", "0.987"),
            ("precision", "0.961"),
            ("recall", "0.943"),
        ],
        "hyperparams": [("max_depth", "8"), ("n_estimators", "400")],
        "custom": {
            "label_definition": "customer_churned_within_30d",
            "label_source_table": f"{DB}.analytics.order_history",
            "label_source_column": "order_status",
            "label_logic": (
                "customer has no order reaching a completed status in the 30 "
                "days following as_of_date"
            ),
            "training_window": "2025-01-01/2025-12-31",
            "last_trained": "2026-02-14",
            "retraining_cadence": "quarterly",
        },
        "deployment": "customer_churn_predictor_prod",
    },
    {
        "name": "demand_forecast",
        "description": (
            "Regression model forecasting 14-day product demand per category "
            "to drive replenishment planning."
        ),
        "version": "1.8.0",
        "type": "gradient_boosted_trees",
        "features": ["order_frequency_90d", "avg_basket_value", "promo_response_rate"],
        "metrics": [("mape", "0.142"), ("rmse", "18.4")],
        "hyperparams": [("max_depth", "6"), ("n_estimators", "250")],
        "custom": {
            "label_definition": "units_sold_next_14d",
            "label_source_table": f"{DB}.order_entry.order_items",
            "label_source_column": "quantity",
            "label_logic": (
                "sum of quantity per product category over the 14 days "
                "following the reference date"
            ),
            "training_window": "2025-06-01/2026-05-31",
            "last_trained": "2026-06-02",
            "retraining_cadence": "monthly",
        },
        "deployment": "demand_forecast_prod",
    },
]


def build_mcps():
    """Return the full list of MetadataChangeProposalWrappers to emit."""
    mcps = []

    # --- features -------------------------------------------------------
    feature_urns = []
    for f in FEATURES:
        urn = str(MlFeatureUrn(FEATURE_TABLE, f["name"]))
        feature_urns.append(urn)
        mcps.append(
            MetadataChangeProposalWrapper(
                entityUrn=urn,
                aspect=sc.MLFeaturePropertiesClass(
                    description=f["description"],
                    dataType=f["type"],
                    sources=f["sources"],
                ),
            )
        )

    pk_urn = str(MlPrimaryKeyUrn(FEATURE_TABLE, PRIMARY_KEY["name"]))
    mcps.append(
        MetadataChangeProposalWrapper(
            entityUrn=pk_urn,
            aspect=sc.MLPrimaryKeyPropertiesClass(
                description=PRIMARY_KEY["description"],
                dataType=PRIMARY_KEY["type"],
                sources=PRIMARY_KEY["sources"],
            ),
        )
    )

    # --- feature table --------------------------------------------------
    table_urn = str(MlFeatureTableUrn(FEATURE_PLATFORM, FEATURE_TABLE))
    mcps.append(
        MetadataChangeProposalWrapper(
            entityUrn=table_urn,
            aspect=sc.MLFeatureTablePropertiesClass(
                description=(
                    "Customer-level features serving churn and demand models "
                    "for the order-entry domain."
                ),
                mlFeatures=feature_urns,
                mlPrimaryKeys=[pk_urn],
            ),
        )
    )

    # --- models and deployments ----------------------------------------
    for m in MODELS:
        model_urn = str(MlModelUrn(MODEL_PLATFORM, m["name"], ENV))
        deploy_urn = str(MlModelDeploymentUrn(MODEL_PLATFORM, m["deployment"], ENV))

        mcps.append(
            MetadataChangeProposalWrapper(
                entityUrn=deploy_urn,
                aspect=sc.MLModelDeploymentPropertiesClass(
                    description=f"Production serving endpoint for {m['name']}.",
                    version=sc.VersionTagClass(versionTag=m["version"]),
                    status=sc.DeploymentStatusClass.IN_SERVICE,
                    createdAt=int(time.time() * 1000),
                ),
            )
        )

        mcps.append(
            MetadataChangeProposalWrapper(
                entityUrn=model_urn,
                aspect=sc.MLModelPropertiesClass(
                    name=m["name"],
                    description=m["description"],
                    type=m["type"],
                    version=sc.VersionTagClass(versionTag=m["version"]),
                    mlFeatures=[
                        str(MlFeatureUrn(FEATURE_TABLE, n)) for n in m["features"]
                    ],
                    deployments=[deploy_urn],
                    trainingMetrics=[
                        sc.MLMetricClass(name=k, value=v) for k, v in m["metrics"]
                    ],
                    hyperParams=[
                        sc.MLHyperParamClass(name=k, value=v)
                        for k, v in m["hyperparams"]
                    ],
                    customProperties=m["custom"],
                ),
            )
        )

    return mcps


def check_datasets(graph) -> bool:
    """Warn if the upstream datasets are missing; lineage would dangle.

    Assumes the caller has already established that the server responds. A
    missing dataset is a warning; an absent server is not, and conflating the
    two produced a wall of traceback for what is really a one-line problem.
    """
    missing = [
        u
        for u in (CUSTOMERS, ORDER_ITEMS, ORDER_HISTORY, PROMOTIONS, ORDER_DETAILS)
        if not graph.exists(u)
    ]
    if missing:
        print("WARNING: these upstream datasets were not found:")
        for u in missing:
            print(f"   {u}")
        print(
            "\nThe datapack may not be loaded, or PREFIX is wrong.\n"
            "Load it with:  datahub datapack load showcase-ecommerce\n"
            "Feature lineage will dangle until they exist.\n"
        )
        return False
    print(f"Upstream datasets found ({len(FEATURES)} features will attach).")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gms", default="http://localhost:8080")
    ap.add_argument("--token", default=None)
    ap.add_argument(
        "--dry-run", action="store_true", help="print what would be emitted, emit nothing"
    )
    args = ap.parse_args()

    mcps = build_mcps()

    if args.dry_run:
        for mcp in mcps:
            print(f"{mcp.entityUrn}\n    {type(mcp.aspect).__name__}")
        print(f"\n{len(mcps)} aspects (dry run, nothing emitted).")
        return 0

    from datahub.ingestion.graph.client import DataHubGraph, DatahubClientConfig

    try:
        graph = DataHubGraph(DatahubClientConfig(server=args.gms, token=args.token))
        graph.test_connection()
    except Exception as e:  # noqa: BLE001
        print(f"Cannot reach DataHub at {args.gms}", file=sys.stderr)
        print(f"  {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        print("\nStart it with: datahub docker quickstart --no-pull-images",
              file=sys.stderr)
        return 1

    check_datasets(graph)

    emitter = DatahubRestEmitter(gms_server=args.gms, token=args.token)
    for mcp in mcps:
        emitter.emit(mcp)

    print(f"\nEmitted {len(mcps)} aspects.")
    print("\nLineage chain now traversable:")
    print("   snowflake datasets -> mlFeature -> mlModel -> mlModelDeployment")
    print(f"\nInspect at {args.gms.replace('8080', '9002')}/mlModels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
