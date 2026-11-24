#!/usr/bin/env python3
"""Benchmark a small set of expert solves before choosing replanning period."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
STAGE4 = RESEARCH / "04_nuplan_static_obstacle"
STAGE5 = RESEARCH / "05_nuplan_ocd_comparison"
sys.path[:0] = [str(STAGE4), str(STAGE5), str(HERE)]

from run_official_interplan import build_scenario, stopped_vehicle_poses  # noqa: E402
from run_comparison import apply_initial_speed_override  # noqa: E402
from nuplan.common.actor_state.ego_state import EgoState  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint  # noqa: E402
from expert_optimizer import ExpertConfig, OptimalControlExpert, SceneGeometry  # noqa: E402


def ego_from_row(row, vehicle):
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(float(row["rear_axle_x_m"]), float(row["rear_axle_y_m"]), float(row["heading_rad"])),
        rear_axle_velocity_2d=StateVector2D(float(row["rear_axle_vx_mps"]), float(row["rear_axle_vy_mps"])),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=float(row["tire_steering_angle_rad"]),
        time_point=TimePoint(int(row["timestamp_us"])),
        vehicle_parameters=vehicle,
        is_in_auto_mode=True,
        angular_vel=float(row["yaw_rate_radps"]),
        angular_accel=0.0,
        tire_steering_rate=0.0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/solver_benchmark.json")
    parser.add_argument(
        "--sample-times", nargs="+", type=float, default=[0.0, 0.0, 0.0, 0.0, 0.0],
        help="Baseline-log states to solve from; the default repeats the common initial state for cold-start timing.",
    )
    args = parser.parse_args()

    scenario, _ = build_scenario("81f400b414215325-s0")
    apply_initial_speed_override(scenario, 15.0)
    obstacle_pose = stopped_vehicle_poses(scenario)[0]
    geometry = SceneGeometry(scenario, obstacle_pose)
    config = ExpertConfig()
    baseline_path = STAGE5 / "results_ocd_peak_sweep_15mps/high_1p0/ego_states.csv"
    with baseline_path.open() as stream:
        baseline = list(csv.DictReader(stream))
    records = []
    for sample_time in args.sample_times:
        row = min(baseline, key=lambda item: abs(float(item["time_s"]) - sample_time))
        ego = ego_from_row(row, scenario.ego_vehicle_parameters)
        # Fresh optimizer per sample prevents warm-start timing from hiding cold-start cost.
        expert = OptimalControlExpert(geometry, config, pass_side=-1)
        result = expert.solve(geometry.initial_vector(ego))
        records.append({
            "requested_time_s": sample_time,
            "actual_time_s": float(row["time_s"]),
            "success": result.success,
            "status": result.status,
            "solve_time_s": result.solve_time_s,
            "iterations": result.iterations,
            "objective": result.objective,
            "message": result.message,
        })
    durations = np.asarray([item["solve_time_s"] for item in records], dtype=float)
    successful = [item for item in records if item["success"]]
    payload = {
        "experiment": "真实 nuPlan 场景，自车初速度设为 15 m/s 的压力测试；15 m/s 是实验设定，不是原日志速度。",
        "samples": records,
        "statistics": {
            "count": len(records),
            "successes": len(successful),
            "failures": len(records) - len(successful),
            "mean_s": float(durations.mean()),
            "p95_s": float(np.percentile(durations, 95)),
            "maximum_s": float(durations.max()),
            "over_0p1_s": int(np.sum(durations > 0.1)),
            "over_0p5_s": int(np.sum(durations > 0.5)),
            "over_1p0_s": int(np.sum(durations > 1.0)),
        },
        "expert": OptimalControlExpert(geometry, config, pass_side=-1).metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
