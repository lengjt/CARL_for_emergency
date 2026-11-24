#!/usr/bin/env python3
"""Collect and screen a three-maneuver pilot expert dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path[:0] = [
    str(RESEARCH / "04_nuplan_static_obstacle"),
    str(RESEARCH / "05_nuplan_ocd_comparison"),
    str(RESEARCH / "06_optimal_control_expert"),
    str(HERE),
]

from run_official_interplan import build_scenario, stopped_vehicle_poses  # noqa: E402
from run_comparison import apply_initial_speed_override  # noqa: E402
from expert_optimizer import SceneGeometry  # noqa: E402
from run_expert_closed_loop import run as run_expert  # noqa: E402
from data_contract import (  # noqa: E402
    CONTRACT_VERSION, build_condition, package_solution, stable_id, write_json,
)


DEFAULT_CONFIG = HERE / "config/pilot.json"
DEFAULT_OUTPUT = HERE / "outputs/pilot"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--candidate", choices=["safe_stop", "pass_left", "pass_right"], action="append")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"Unsupported contract: {config['contract_version']}")

    output = args.output_root.resolve()
    raw_root = output / "raw_runs"
    dataset_root = output / "dataset"
    scenario, _ = build_scenario(config["scenario_token"])
    apply_initial_speed_override(scenario, config["initial_speed_mps"])
    geometry = SceneGeometry(scenario, stopped_vehicle_poses(scenario)[0])
    condition = build_condition(scenario, geometry, config)
    condition_id = stable_id("condition", condition)
    condition["condition_id"] = condition_id
    write_json(dataset_root / "conditions" / condition_id / "condition.json", condition)

    selected_names = set(args.candidate or [item["maneuver_type"] for item in config["candidates"]])
    manifests = []
    for candidate in config["candidates"]:
        maneuver_type = candidate["maneuver_type"]
        if maneuver_type not in selected_names:
            solution_id = stable_id("solution", {"condition_id": condition_id, "maneuver_type": maneuver_type})
            existing = dataset_root / "solutions" / solution_id / "solution.json"
            if existing.is_file():
                manifests.append(json.loads(existing.read_text()))
            continue
        solution_id = stable_id("solution", {"condition_id": condition_id, "maneuver_type": maneuver_type})
        print(f"[{maneuver_type}] running {solution_id}", flush=True)
        try:
            pass_side = 1 if candidate["pass_side"] == "left" else -1
            run_expert(
                period_s=config["expert_replan_period_s"],
                force=args.force,
                output_root=raw_root,
                output_name=solution_id,
                initial_speed_mps=config["initial_speed_mps"],
                peak_scale=config["ocd_internal_mf52_peak_scale"],
                maneuver=candidate["optimizer_maneuver"],
                pass_side=pass_side,
                scenario_token=config["scenario_token"],
            )
            manifest = package_solution(
                raw_root / solution_id,
                dataset_root / "solutions" / solution_id,
                condition_id,
                condition,
                maneuver_type,
                scenario,
                geometry,
            )
        except Exception as error:  # retain candidate-level failures and continue screening
            manifest = {
                "contract_version": CONTRACT_VERSION,
                "condition_id": condition_id,
                "solution_id": solution_id,
                "maneuver_type": maneuver_type,
                "admitted_as_training_label": False,
                "rejection_reasons": ["collection_exception"],
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            write_json(dataset_root / "solutions" / solution_id / "solution.json", manifest)
        manifests.append(manifest)
        print(json.dumps({
            "maneuver_type": maneuver_type,
            "admitted": manifest["admitted_as_training_label"],
            "rejection_reasons": manifest["rejection_reasons"],
        }), flush=True)

    dataset_manifest = {
        "contract_version": CONTRACT_VERSION,
        "condition_ids": [condition_id],
        "solutions": manifests,
        "accepted_solution_ids": [item["solution_id"] for item in manifests if item["admitted_as_training_label"]],
        "rejected_solution_ids": [item["solution_id"] for item in manifests if not item["admitted_as_training_label"]],
        "policy": "same-condition alternatives are retained only after independent LQR+OCD validation",
    }
    write_json(dataset_root / "dataset_manifest.json", dataset_manifest)
    print(json.dumps(dataset_manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
