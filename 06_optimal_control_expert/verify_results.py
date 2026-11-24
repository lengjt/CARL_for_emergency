#!/usr/bin/env python3
"""Independent structural checks for the frozen first-stage comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "05_nuplan_ocd_comparison/results_ocd_peak_sweep_15mps/high_1p0"
EXPERT = HERE / "results/replan_0p1s"


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    baseline_summary = json.loads((BASELINE / "summary.json").read_text())
    expert_summary = json.loads((EXPERT / "summary.json").read_text())
    baseline_states, expert_states = rows(BASELINE / "ego_states.csv"), rows(EXPERT / "ego_states.csv")
    baseline_references = np.load(BASELINE / "reference_trajectories_global.npz")["trajectories"]
    expert_references = np.load(EXPERT / "reference_trajectories_global.npz")["trajectories"]
    baseline_obstacles = np.load(BASELINE / "stopped_vehicle_tracks.npz")["vehicle_0"]
    expert_obstacles = np.load(EXPERT / "stopped_vehicle_tracks.npz")["vehicle_0"]
    checks = {
        "same_initial_timestamp": baseline_states[0]["timestamp_us"] == expert_states[0]["timestamp_us"],
        "same_initial_pose": max(abs(float(baseline_states[0][key]) - float(expert_states[0][key]))
                                 for key in ("rear_axle_x_m", "rear_axle_y_m", "heading_rad")) < 1e-10,
        "same_initial_speed_15_mps": abs(float(baseline_states[0]["speed_mps"]) - 15.0) < 1e-12
        and abs(float(expert_states[0]["speed_mps"]) - 15.0) < 1e-12,
        "same_181_state_horizon": len(baseline_states) == len(expert_states) == 181,
        "same_reference_shape": baseline_references.shape == expert_references.shape == (181, 81, 5),
        "same_static_obstacle": baseline_obstacles.shape == expert_obstacles.shape
        and float(np.max(np.abs(baseline_obstacles - expert_obstacles))) < 1e-10,
        "same_ocd_peak_scale": baseline_summary["ocd_vehicle"]["internal_peak_scale"]
        == expert_summary["ocd_vehicle"]["internal_peak_scale"] == 1.0,
        "same_lqr_ocd_update_count": len(rows(BASELINE / "lqr_controls.csv"))
        == len(rows(EXPERT / "lqr_controls.csv")) == 180,
        "expert_uses_trajectory_interface": expert_summary["reference_interface"]
        == "time-parameterized nuPlan InterpolatedTrajectory",
        "expert_does_not_control_ocd_directly": expert_summary["direct_control_to_ocd"] is False,
        "frozen_period_is_0p1_s": expert_summary["replan_period_s"] == 0.1,
        "expert_closed_loop_safe": expert_summary["first_collision_time_s"] is None
        and expert_summary["first_offroad_time_s"] is None,
    }
    payload = {"passed": all(checks.values()), "checks": checks}
    (HERE / "results/comparison/verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
