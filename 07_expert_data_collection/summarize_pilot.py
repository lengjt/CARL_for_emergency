#!/usr/bin/env python3
"""Write a compact screening report for the pilot collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "outputs/pilot/dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    rows = []
    for item in manifest["solutions"]:
        metrics = item.get("closed_loop_metrics", {})
        rows.append({
            "maneuver_type": item["maneuver_type"],
            "admitted": item["admitted_as_training_label"],
            "fresh_labels": item.get("fresh_reference_label_count", 0),
            "collision": metrics.get("collision"),
            "road_departure": metrics.get("road_departure"),
            "minimum_clearance_m": metrics.get("minimum_obstacle_clearance_m"),
            "passed_obstacle": metrics.get("passed_obstacle"),
            "reached_safe_stop": metrics.get("reached_safe_stop"),
            "detected_pass_side": metrics.get("detected_pass_side"),
            "rejection_reasons": item.get("rejection_reasons", []),
        })
    payload = {
        "contract_version": manifest["contract_version"],
        "condition_id": manifest["condition_ids"][0],
        "accepted_solution_count": len(manifest["accepted_solution_ids"]),
        "rejected_solution_count": len(manifest["rejected_solution_ids"]),
        "solutions": rows,
    }
    (root / "screening_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    lines = [
        "# Pilot expert-data screening", "",
        f"Condition: `{payload['condition_id']}`", "",
        "| Maneuver | Admitted | Fresh labels | Collision | Off-road | Clearance (m) | Actual side | Reason |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        clearance = "-" if row["minimum_clearance_m"] is None else f"{row['minimum_clearance_m']:.3f}"
        lines.append(
            f"| {row['maneuver_type']} | {row['admitted']} | {row['fresh_labels']} | "
            f"{row['collision']} | {row['road_departure']} | {clearance} | "
            f"{row['detected_pass_side'] or '-'} | {', '.join(row['rejection_reasons']) or '-'} |"
        )
    lines.extend([
        "", "Only admitted solutions are positive training labels. Reference labels and executed trajectories are stored separately.",
    ])
    (root / "SCREENING_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
