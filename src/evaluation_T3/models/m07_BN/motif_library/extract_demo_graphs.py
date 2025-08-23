#!/usr/bin/env python3
"""
Extract agent demographics and graphData pairs from survey datasets and save as per-agent JSON files.

Joins:
  - responses_*.json (array) providing { id, agent } demographics
  - causal_graph_responses_*.json (array) providing { prolificId, graphs: [{ qaPairId, graphData }] }

We link by id == prolificId. If an agent has no graphs, skip.
"""

import os
import json
import argparse
from typing import Dict, Any
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Extract joined demographics + graphs")
    parser.add_argument(
        "--responses",
        required=True,
        help="Path to responses_*.json with agent demographics",
    )
    parser.add_argument(
        "--graphs",
        required=True,
        help="Path to causal_graph_responses_*.json with graphData",
    )
    args = parser.parse_args()

    # Load responses and build map id -> demographics
    with open(args.responses, "r") as f:
        responses = json.load(f)

    id_to_demo: Dict[str, Dict[str, Any]] = {}
    for entry in responses:
        rid = entry.get("id")
        if not rid:
            continue
        id_to_demo[rid] = entry.get("agent", {})

    # Load graphs array
    with open(args.graphs, "r") as f:
        graph_entries = json.load(f)

    written_agents = 0
    skipped = 0

    # Aggregate per agent into a single structure
    id_to_record: Dict[str, Any] = {}

    for respondent in graph_entries:
        pid = respondent.get("prolificId")
        if not pid:
            continue
        demo = id_to_demo.get(pid)
        graphs = respondent.get("graphs", []) or []
        if not demo or not graphs:
            skipped += 1
            continue

        # Pick the latest graph by timestamp; fallback to last in list
        def _extract_ts(g: Dict[str, Any]) -> float:
            ts = g.get("timestamp")
            if ts is None:
                ts = g.get("graphData", {}).get("timestamp")
            # Normalize
            if isinstance(ts, (int, float)):
                return float(ts)
            if isinstance(ts, str):
                try:
                    # Handle trailing Z by replacing with +00:00 for fromisoformat
                    iso = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
                    return datetime.fromisoformat(iso).timestamp()
                except Exception:
                    return -1.0
            return -1.0

        latest_graph = None
        latest_ts = -1.0
        for g in graphs:
            ts_val = _extract_ts(g)
            if ts_val >= latest_ts:
                latest_ts = ts_val
                latest_graph = g

        if latest_graph is None:
            latest_graph = graphs[-1]

        rec = id_to_record.get(pid)
        if not rec:
            rec = {"agent_id": pid, "demographics": demo, "graphs": []}
            id_to_record[pid] = rec
            written_agents += 1

        qa_id = latest_graph.get("qaPairId") or latest_graph.get("graphData", {}).get(
            "qaPairId"
        )
        graph_data = latest_graph.get("graphData", {})
        rec["graphs"] = [{"qaPairId": qa_id, "graph": graph_data}]

    # Decide output location and filename
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "agents_with_graphs.json")

    with open(output_path, "w") as fout:
        json.dump(list(id_to_record.values()), fout, indent=2)

    total_graphs = sum(len(r.get("graphs", [])) for r in id_to_record.values())
    print(
        f"Written file: {output_path}. Agents: {written_agents}, Graphs: {total_graphs}, Skipped agents: {skipped}"
    )


if __name__ == "__main__":
    main()
