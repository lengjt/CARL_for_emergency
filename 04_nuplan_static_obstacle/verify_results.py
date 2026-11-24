#!/usr/bin/env python3
"""Independent consistency checks over the two saved experiment logs."""
import csv
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CFG = yaml.safe_load((HERE / "config.yaml").read_text())


def rows(variant):
    with (RESULTS / variant / "ego_states.csv").open() as f:
        return list(csv.DictReader(f))


def angle_error(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def main():
    global CFG, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = parser.parse_args()
    CFG = yaml.safe_load(args.config.resolve().read_text())
    RESULTS = Path(CFG["project"].get("results_dir", HERE / "results"))
    baseline = rows("baseline")
    obstacle = rows("obstacle")
    bsum = json.loads((RESULTS / "baseline" / "summary.json").read_text())
    osum = json.loads((RESULTS / "obstacle" / "summary.json").read_text())
    geometry = json.loads((RESULTS / "obstacle" / "obstacle_geometry.json").read_text())
    audits = json.loads((RESULTS / "obstacle" / "model_input_audit.json").read_text())

    initial_fields = ["rear_axle_x_m", "rear_axle_y_m", "heading_rad", "speed_mps"]
    initial_errors = {key: abs(float(baseline[0][key]) - float(obstacle[0][key])) for key in initial_fields}
    timestamp_equal = [row["timestamp_us"] for row in baseline] == [row["timestamp_us"] for row in obstacle]
    reconstructed_position_errors = []
    reconstructed_heading_errors = []
    reconstructed_dimension_errors = []
    reconstructed_speed_errors = []
    dynamic_agent = CFG["obstacle"].get("representation", "static_object") == "dynamic_agent"
    for row, audit in zip(obstacle, audits):
        slot = audit["obstacle_agent_slot"] if dynamic_agent else audit["obstacle_slot"]
        if slot is None:
            continue
        feature = audit["agent_current_tensor"][slot] if dynamic_agent else audit["static_tensor"][slot]
        ego_x, ego_y, ego_h = float(row["rear_axle_x_m"]), float(row["rear_axle_y_m"]), float(row["heading_rad"])
        c, s = math.cos(ego_h), math.sin(ego_h)
        global_x = ego_x + feature[0] * c - feature[1] * s
        global_y = ego_y + feature[0] * s + feature[1] * c
        global_h = ego_h + math.atan2(feature[3], feature[2])
        reconstructed_position_errors.append(math.hypot(global_x - geometry["center_x_m"], global_y - geometry["center_y_m"]))
        reconstructed_heading_errors.append(angle_error(global_h, geometry["heading_rad"]))
        width_index, length_index = (6, 7) if dynamic_agent else (4, 5)
        reconstructed_dimension_errors.extend(
            [abs(feature[width_index] - geometry["width_m"]), abs(feature[length_index] - geometry["length_m"])]
        )
        if dynamic_agent:
            reconstructed_speed_errors.append(math.hypot(feature[4], feature[5]) - geometry["speed_mps"])

    reference_files_ok = True
    reference_shapes = {}
    for variant in ("baseline", "obstacle"):
        data = np.load(RESULTS / variant / "reference_trajectories_global.npz")
        reference_shapes[variant] = list(data["trajectories"].shape)
        reference_files_ok &= data["trajectories"].shape[0] == len(rows(variant))
        reference_files_ok &= data["trajectories"].shape[2] == 5

    checks = {
        "same_initial_state": max(initial_errors.values()) < 1e-12,
        "same_timestamps": timestamp_equal,
        "same_checkpoint": bsum["checkpoint_sha256"] == osum["checkpoint_sha256"],
        "same_seed_and_sampling": (bsum["seed"], bsum["cfg_weight"], bsum["sample_steps"])
        == (osum["seed"], osum["cfg_weight"], osum["sample_steps"]),
        "obstacle_seen_every_planner_call": all(
            a["obstacle_agent_slot" if dynamic_agent else "obstacle_slot"] is not None for a in audits
        ),
        "obstacle_geometry_matches_config": (
            geometry["category"] == CFG["obstacle"]["category"]
            and abs(geometry["length_m"] - CFG["obstacle"]["length_m"]) < 1e-12
            and abs(geometry["width_m"] - CFG["obstacle"]["width_m"]) < 1e-12
            and abs(
                geometry["heading_rad"]
                - (
                    CFG["obstacle"]["heading_rad"]
                    if CFG["obstacle"].get("placement", "relative_initial_pose") == "global"
                    else float(obstacle[0]["heading_rad"]) + CFG["obstacle"]["heading_offset_rad"]
                )
            ) < 1e-12
        ),
        "model_tensor_reconstructs_collision_geometry": max(reconstructed_position_errors) < 1e-4
        and max(reconstructed_heading_errors) < 1e-5
        and max(reconstructed_dimension_errors) < 1e-5
        and (not dynamic_agent or max(abs(x) for x in reconstructed_speed_errors) < 1e-5),
        "reference_and_actual_global_frame_consistent": max(
            bsum["max_reference_current_position_error_m"], osum["max_reference_current_position_error_m"]
        ) < 1e-9,
        "full_reference_logs_present": bool(reference_files_ok),
        "actual_model_execution_recorded": (
            bsum["lqr_control_count"] == len(baseline) - 1
            and osum["lqr_control_count"] == len(obstacle) - 1
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "initial_state_absolute_errors": initial_errors,
        "max_model_tensor_to_obstacle_position_error_m": max(reconstructed_position_errors),
        "max_model_tensor_to_obstacle_heading_error_rad": max(reconstructed_heading_errors),
        "max_model_tensor_to_obstacle_dimension_error_m": max(reconstructed_dimension_errors),
        "max_model_tensor_to_obstacle_speed_error_mps": max((abs(x) for x in reconstructed_speed_errors), default=0.0),
        "reference_array_shapes": reference_shapes,
    }
    (RESULTS / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

