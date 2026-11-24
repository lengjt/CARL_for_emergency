#!/usr/bin/env python3
"""Summarize obstacle response from the saved, already-run logs."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config_wide_lane.yaml").read_text())
RESULTS = Path(CFG["project"]["results_dir"])


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def numeric(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def main():
    obstacle = CFG["obstacle"]
    heading = float(obstacle["heading_rad"])
    forward = np.asarray([math.cos(heading), math.sin(heading)])
    left = np.asarray([-math.sin(heading), math.cos(heading)])
    center = np.asarray([obstacle["center_x_m"], obstacle["center_y_m"]])
    result = {}
    logs = {}
    for variant in ("baseline", "obstacle"):
        rows = read_csv(RESULTS / variant / "ego_states.csv")
        xy = np.column_stack([numeric(rows, "rear_axle_x_m"), numeric(rows, "rear_axle_y_m")])
        delta = xy - center
        logs[variant] = {
            "rows": rows,
            "time": numeric(rows, "time_s"),
            "speed": numeric(rows, "speed_mps"),
            "longitudinal": delta @ forward,
            "lateral": delta @ left,
        }
    collision_indices = np.flatnonzero([int(row["obstacle_collision"]) for row in logs["obstacle"]["rows"]])
    first_collision = int(collision_indices[0]) if len(collision_indices) else None
    result["first_collision"] = None if first_collision is None else {
        "iteration": first_collision,
        "time_s": float(logs["obstacle"]["time"][first_collision]),
        "speed_mps": float(logs["obstacle"]["speed"][first_collision]),
        "rear_axle_longitudinal_from_barrier_m": float(logs["obstacle"]["longitudinal"][first_collision]),
        "rear_axle_lateral_from_barrier_m": float(logs["obstacle"]["lateral"][first_collision]),
    }
    sample_times = [0, 1, 2, 3, 4, 4.6]
    result["actual_speed_comparison"] = []
    for requested in sample_times:
        idx = int(np.argmin(abs(logs["obstacle"]["time"] - requested)))
        result["actual_speed_comparison"].append({
            "time_s": float(logs["obstacle"]["time"][idx]),
            "baseline_mps": float(logs["baseline"]["speed"][idx]),
            "obstacle_mps": float(logs["obstacle"]["speed"][idx]),
            "difference_mps": float(logs["obstacle"]["speed"][idx] - logs["baseline"]["speed"][idx]),
        })
    required = (2.297 + float(obstacle["width_m"])) / 2
    result["parallel_center_offset_required_to_clear_m"] = required
    result["reference_at_barrier"] = {}
    for variant in ("baseline", "obstacle"):
        refs = np.load(RESULTS / variant / "reference_trajectories_global.npz")["trajectories"]
        entries = []
        for requested in [0, 1, 2, 3, 4]:
            call = int(np.argmin(abs(logs[variant]["time"] - requested)))
            delta = refs[call, :, 1:3] - center
            longitudinal = delta @ forward
            crossing = int(np.argmin(abs(longitudinal)))
            lateral = float(delta[crossing] @ left)
            entries.append({
                "planning_time_s": float(logs[variant]["time"][call]),
                "reference_time_ahead_s": float((refs[call, crossing, 0] - refs[call, 0, 0]) * 1e-6),
                "lateral_center_offset_m": lateral,
                "clearance_margin_m": abs(lateral) - required,
                "planned_speed_mps": float(refs[call, crossing, 4]),
            })
        result["reference_at_barrier"][variant] = entries
    for variant in ("baseline", "obstacle"):
        controls = read_csv(RESULTS / variant / "lqr_controls.csv")
        accel = numeric(controls, "acceleration_mps2")
        time = (numeric(controls, "timestamp_us") - numeric(controls, "timestamp_us")[0]) * 1e-6
        mask = time < (result["first_collision"]["time_s"] if result["first_collision"] else 5.0)
        result.setdefault("precollision_lqr", {})[variant] = {
            "mean_acceleration_mps2": float(accel[mask].mean()),
            "minimum_acceleration_mps2": float(accel[mask].min()),
            "maximum_acceleration_mps2": float(accel[mask].max()),
        }
    output = RESULTS / "behavior_analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
