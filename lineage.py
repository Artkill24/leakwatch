#!/usr/bin/env python3
"""
Walk column-level lineage upward from a target column.

Why this exists
---------------
A label is defined on one column of one table. The same information often lives
upstream under a different address: a curated table and the raw table it was
built from both carry `order_status`, and a feature sourced from the raw one is
reading the label without any description saying so.

No amount of reading feature descriptions finds that. It is a property of the
graph, so it is resolved in the graph.

Cycles are not hypothetical here: in the showcase-ecommerce datapack,
`analytics.order_history` lists itself among its own upstreams. Any recursive
walk without a visited set hangs on the first call.

Usage:
    python lineage.py --table b2fd91.order_entry_db.analytics.order_history \\
                      --column order_status
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Set, Tuple

from datahub.metadata.urns import SchemaFieldUrn
import datahub.metadata.schema_classes as sc

# (dataset_urn, column_name)
ColumnRef = Tuple[str, str]

MAX_DEPTH = 6


def _canonical(dataset_urn: str) -> str:
    """Case-fold the dataset name so ORDER_ENTRY_DB and order_entry_db collapse.

    The datapack carries both spellings of the same table, and without this the
    target ends up listed among its own ancestors.
    """
    return dataset_urn.lower()


def _split_field_urn(urn: str) -> ColumnRef | None:
    try:
        parsed = SchemaFieldUrn.from_string(urn)
        return (str(parsed.parent), parsed.field_path)
    except Exception:  # noqa: BLE001
        return None


def column_ancestry(
    graph,
    dataset_urn: str,
    column: str,
    max_depth: int = MAX_DEPTH,
) -> Dict[ColumnRef, int]:
    """Every column that feeds the given one, mapped to its distance in hops.

    Distance matters: a direct upstream is stronger evidence than something four
    hops away, and the caller may want to weight accordingly.
    """
    found: Dict[ColumnRef, int] = {}
    origin = (_canonical(dataset_urn), column.lower())
    visited: Set[ColumnRef] = {origin}
    frontier: List[Tuple[ColumnRef, int]] = [((dataset_urn, column), 0)]

    while frontier:
        (ds, col), depth = frontier.pop()
        if depth >= max_depth:
            continue

        lineage = graph.get_aspect(ds, sc.UpstreamLineageClass)
        if not lineage or not lineage.fineGrainedLineages:
            continue

        for fg in lineage.fineGrainedLineages:
            downstream_cols = {
                ref[1]
                for ref in (_split_field_urn(d) for d in (fg.downstreams or []))
                if ref
            }
            if col not in downstream_cols:
                continue

            for up_urn in fg.upstreams or []:
                ref = _split_field_urn(up_urn)
                if not ref:
                    continue
                canon = (_canonical(ref[0]), ref[1].lower())
                # skip the origin reached under a different spelling, and any
                # column already seen: the datapack has self-referential lineage
                if canon == origin or canon in visited:
                    continue
                visited.add(canon)
                found[ref] = depth + 1
                frontier.append((ref, depth + 1))

    return found


def tables_carrying_target(
    graph, dataset_urn: str, column: str, max_depth: int = MAX_DEPTH
) -> Dict[str, Tuple[str, int]]:
    """Collapse the ancestry to: dataset urn -> (column there, hops away).

    This is what the rule engine consumes. A feature sourcing any of these
    tables is reading a table that carries the label's information, whatever its
    description claims.
    """
    out: Dict[str, Tuple[str, int]] = {}
    for (ds, col), depth in column_ancestry(graph, dataset_urn, column, max_depth).items():
        if ds not in out or depth < out[ds][1]:
            out[ds] = (col, depth)
    return out


def target_column_in_tables(
    graph, target_table_urn: str, column: str, candidate_tables: set
) -> Dict[str, int]:
    """For each candidate table, at what hop does it carry the target column?

    hop 0 means the candidate *is* the target's table -- the feature reads the
    label column directly. Higher hops mean it reads the column further upstream.
    Returns only tables that carry it; absent tables don't.

    This replaces asking the LLM "does this mention the target column", which was
    a judgement dressed as a fact and oscillated between runs. Table membership
    is a graph property and returns the same answer every time.
    """
    out: Dict[str, int] = {}
    target_short = target_table_urn.split(",")[1] if "," in target_table_urn else target_table_urn

    # hop 0: the target's own table
    for cand in candidate_tables:
        cand_short = cand.split(",")[1] if "," in cand else cand
        if cand_short.lower() == target_short.lower():
            out[cand] = 0

    # hop >=1: ancestors carrying the column
    for ds, (col, depth) in tables_carrying_target(graph, target_table_urn, column).items():
        ds_short = ds.split(",")[1] if "," in ds else ds
        for cand in candidate_tables:
            cand_short = cand.split(",")[1] if "," in cand else cand
            if cand_short.lower() == ds_short.lower():
                if cand not in out or depth < out[cand]:
                    out[cand] = depth
    return out


def _platform(dataset_urn: str) -> str:
    try:
        return dataset_urn.split("dataPlatform:")[1].split(",")[0]
    except IndexError:
        return "?"


def _short(dataset_urn: str) -> str:
    return dataset_urn.split(",")[1] if "," in dataset_urn else dataset_urn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gms", default="http://localhost:8080")
    ap.add_argument("--table", required=True, help="dataset name, not urn")
    ap.add_argument("--column", required=True)
    ap.add_argument("--platform", default="snowflake")
    ap.add_argument("--env", default="PROD")
    args = ap.parse_args()

    from context import MLContextExtractor

    try:
        ex = MLContextExtractor(args.gms)
        ex.graph.test_connection()
    except Exception as e:  # noqa: BLE001
        print(f"Cannot reach DataHub: {type(e).__name__}", file=sys.stderr)
        return 1

    urn = (
        f"urn:li:dataset:(urn:li:dataPlatform:{args.platform},"
        f"{args.table},{args.env})"
    )

    print(f"target: {args.table}.{args.column}\n")

    ancestry = column_ancestry(ex.graph, urn, args.column)
    if not ancestry:
        print("no column-level ancestry (passthrough lineage may still exist "
              "at table level)")
        return 0

    for (ds, col), depth in sorted(ancestry.items(), key=lambda x: x[1]):
        print(f"  {depth} hop  [{_platform(ds):9}] {_short(ds)}.{col}")

    print("\ntables carrying the target's information:")
    for ds, (col, depth) in sorted(
        tables_carrying_target(ex.graph, urn, args.column).items(),
        key=lambda x: x[1][1],
    ):
        print(f"  [{_platform(ds):9}] {_short(ds)}  (as {col}, {depth} hop)")

    print("\nNote: the same logical table appears once per platform. A dbt model\n"
          "and the warehouse table it produces are distinct entities in the\n"
          "catalog, and both legitimately carry the column.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
