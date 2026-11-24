#!/usr/bin/env python3
"""Verify structural and semantic invariants of the pilot expert dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "outputs/pilot/dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    checks = {
        "contract_version": manifest["contract_version"] == "expert_dataset_v1",
        "one_shared_condition": len(manifest["condition_ids"]) == 1,
        "all_three_candidates_screened": {item["maneuver_type"] for item in manifest["solutions"]}
        == {"safe_stop", "pass_left", "pass_right"},
        "at_least_one_admitted_solution": bool(manifest["accepted_solution_ids"]),
    }
    for solution in manifest["solutions"]:
        solution_id = solution["solution_id"]
        directory = root / "solutions" / solution_id
        checks[f"{solution_id}:manifest_exists"] = (directory / "solution.json").is_file()
        if not solution["admitted_as_training_label"]:
            continue
        labels = np.load(directory / "reference_labels.npz")
        executed = np.load(directory / "executed_trajectory.npz")
        count = solution["fresh_reference_label_count"]
        checks[f"{solution_id}:fresh_labels_nonempty"] = count > 0
        checks[f"{solution_id}:labels_aligned"] = (
            labels["reference_local"].shape == (count, 81, 5)
            and labels["reference_global"].shape == (count, 81, 5)
            and labels["query_state_global"].shape == (count, 7)
            and labels["dynamics_condition"].shape == (count, 5)
        )
        checks[f"{solution_id}:relative_time_grid"] = bool(np.allclose(
            labels["reference_local"][:, :, 0], np.arange(81) * 0.1, atol=1e-6
        ))
        checks[f"{solution_id}:reference_starts_at_query"] = bool(np.max(
            np.abs(labels["reference_local"][:, 0, 1:4])
        ) < 1e-5)
        checks[f"{solution_id}:execution_separate"] = executed["trajectory"].ndim == 2
        checks[f"{solution_id}:closed_loop_safe"] = (
            not solution["closed_loop_metrics"]["collision"]
            and not solution["closed_loop_metrics"]["road_departure"]
        )
        if solution["maneuver_type"].startswith("pass_"):
            checks[f"{solution_id}:pass_side_matches"] = (
                solution["closed_loop_metrics"]["detected_pass_side"]
                == solution["maneuver_type"].removeprefix("pass_")
            )
    verification = {"passed": bool(all(checks.values())), "checks": checks}
    (root / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))
    if not verification["passed"]:
        raise RuntimeError("Pilot dataset verification failed")


if __name__ == "__main__":
    main()
