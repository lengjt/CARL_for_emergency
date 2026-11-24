#!/usr/bin/env python3
"""Compute compact behavior metrics from the saved final closed-loop logs."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from run_experiment import build_scenario, load_config, make_obstacle
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import StateSE2


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    root = Path(cfg["project"]["results_dir"])
    scenario, _ = build_scenario(cfg)
    rows = {name: read_csv(root / name / "ego_states.csv") for name in ("baseline", "obstacle")}
    controls = {name: read_csv(root / name / "lqr_controls.csv") for name in ("baseline", "obstacle")}

    x0 = float(rows["baseline"][0]["rear_axle_x_m"])
    y0 = float(rows["baseline"][0]["rear_axle_y_m"])
    h0 = float(rows["baseline"][0]["heading_rad"])
    c, s = math.cos(h0), math.sin(h0)

    def local_path(name):
        x = np.asarray([float(row["rear_axle_x_m"]) for row in rows[name]]) - x0
        y = np.asarray([float(row["rear_axle_y_m"]) for row in rows[name]]) - y0
        return c * x + s * y, -s * x + c * y

    local = {name: local_path(name) for name in rows}
    obstacle = make_obstacle(scenario.initial_ego_state, cfg["obstacle"], scenario.start_time.time_us)
    clearance = {}
    for name in rows:
        distances = []
        for row in rows[name]:
            footprint = CarFootprint.build_from_rear_axle(
                StateSE2(
                    float(row["rear_axle_x_m"]),
                    float(row["rear_axle_y_m"]),
                    float(row["heading_rad"]),
                ),
                scenario.ego_vehicle_parameters,
            )
            distances.append(footprint.geometry.distance(obstacle.box.geometry))
        clearance[name] = float(min(distances))

    speed = {
        name: np.asarray([float(row["speed_mps"]) for row in rows[name]]) for name in rows
    }
    acceleration = {
        name: np.asarray([float(row["acceleration_mps2"]) for row in controls[name]]) for name in controls
    }
    obstacle_longitudinal = c * (obstacle.center.x - x0) + s * (obstacle.center.y - y0)
    pass_index = {
        name: int(np.argmin(np.abs(local[name][0] - obstacle_longitudinal))) for name in rows
    }
    lat_difference = local["obstacle"][1] - local["baseline"][1]
    metrics = {
        "obstacle_longitudinal_from_start_m": obstacle_longitudinal,
        "obstacle_lateral_from_start_axis_m": -s * (obstacle.center.x - x0) + c * (obstacle.center.y - y0),
        "baseline_would_collide_with_added_vehicle": clearance["baseline"] == 0.0,
        "obstacle_run_collision_free": not any(int(row["obstacle_collision"]) for row in rows["obstacle"]),
        "minimum_full_body_clearance_m": clearance,
        "lateral_difference_obstacle_minus_baseline_at_pass_m": float(lat_difference[pass_index["obstacle"]]),
        "maximum_absolute_lateral_difference_m": float(np.max(np.abs(lat_difference))),
        "minimum_speed_mps": {name: float(np.min(value)) for name, value in speed.items()},
        "final_speed_mps": {name: float(value[-1]) for name, value in speed.items()},
        "mean_acceleration_first_4s_mps2": {
            name: float(np.mean(value[:40])) for name, value in acceleration.items()
        },
        "model_input_seen_calls": json.loads((root / "obstacle" / "summary.json").read_text())[
            "obstacle_seen_in_model_input_calls"
        ],
        "planner_calls": len(rows["obstacle"]),
    }
    (root / "behavior_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
