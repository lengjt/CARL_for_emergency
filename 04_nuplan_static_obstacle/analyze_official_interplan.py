#!/usr/bin/env python3
"""Aggregate and independently verify the official interPlan closed-loop runs."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results_official_interplan"


def load_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def behavior(summary):
    if summary["first_any_collision_time_s"] is not None:
        return "collision"
    if summary["final_speed_mps"] < 0.5:
        return "safe_stop"
    if summary["minimum_speed_mps"] < 1.0:
        return "slow_or_stop_then_go"
    return "moving_avoidance"


def main():
    scenario_dirs = sorted(path.parent for path in ROOT.glob("*/summary.json"))
    records = []
    verification = {}
    for directory in scenario_dirs:
        summary = json.loads((directory / "summary.json").read_text())
        rows = load_csv(directory / "ego_states.csv")
        controls = load_csv(directory / "lqr_controls.csv")
        audits = json.loads((directory / "model_input_audit.json").read_text())
        references = np.load(directory / "reference_trajectories_global.npz")["trajectories"]
        poses = summary["stopped_vehicle_poses"]
        distances = [
            min(
                math.hypot(float(row["rear_axle_x_m"]) - pose[0], float(row["rear_axle_y_m"]) - pose[1])
                for pose in poses
            )
            for row in rows
        ]
        closest_index = int(np.argmin(distances))
        visible_through_closest = all(
            all(item["slot"] is not None for item in audit["stopped_vehicles"])
            for audit in audits[: closest_index + 1]
        )
        track_data = np.load(directory / "stopped_vehicle_tracks.npz")
        max_motion, max_speed = 0.0, 0.0
        expected_xy = np.asarray([[pose[0], pose[1]] for pose in poses])
        for name in track_data.files:
            track = track_data[name]
            if len(track):
                # The logger can swap indices when there are multiple identical zero-speed
                # vehicles; validate every recorded pose against the set of configured poses.
                pose_errors = np.min(
                    np.linalg.norm(track[:, None, 1:3] - expected_xy[None, :, :], axis=2), axis=1
                )
                max_motion = max(max_motion, float(np.max(pose_errors)))
                max_speed = max(max_speed, float(np.max(track[:, 4])))
        checks = {
            "complete_state_log": len(rows) == summary["frames"],
            "complete_reference_log": references.shape == (summary["planner_calls"], 81, 5),
            "complete_control_log": len(controls) == len(rows) - 1,
            "stopped_objects_static": max_motion < 1e-6 and max_speed < 1e-6,
            "stopped_objects_visible_until_closest_approach": visible_through_closest,
            "released_checkpoint": summary["checkpoint_sha256"]
            == "9231b4c02968c9e42dd2881ad3e7dd4bd362796f7b2c0158e0458c4ab3bde88e",
            "released_sampling": summary["seed"] == 0 and summary["cfg_weight"] == 1.0 and summary["sample_steps"] == 4,
        }
        verification[summary["modified_token"]] = {
            "passed": all(checks.values()),
            "checks": checks,
            "closest_approach_iteration": closest_index,
            "max_stopped_vehicle_motion_m": max_motion,
            "max_stopped_vehicle_speed_mps": max_speed,
        }
        records.append(
            {
                **summary,
                "behavior": behavior(summary),
                "closest_center_distance_m": min(distances),
                "visible_through_closest_approach": visible_through_closest,
                "verification_passed": all(checks.values()),
            }
        )

    by_type = defaultdict(list)
    for record in records:
        by_type[record["scenario_type"]].append(record)
    category_summary = {}
    for scenario_type, items in sorted(by_type.items()):
        modes = Counter(item["behavior"] for item in items)
        category_summary[scenario_type] = {
            "count": len(items),
            "collision_free": sum(item["first_any_collision_time_s"] is None for item in items),
            "collisions": sum(item["first_any_collision_time_s"] is not None for item in items),
            "behavior_counts": dict(sorted(modes.items())),
            "tokens_by_behavior": {
                mode: [item["modified_token"] for item in items if item["behavior"] == mode]
                for mode in sorted(modes)
            },
        }
    aggregate = {
        "scenario_count": len(records),
        "all_verification_passed": all(item["passed"] for item in verification.values()),
        "categories": category_summary,
    }
    (ROOT / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    with (ROOT / "scenario_results.csv").open("w", newline="") as f:
        fields = [
            "modified_token", "scenario_type", "map_name", "behavior", "initial_speed_mps",
            "minimum_speed_mps", "final_speed_mps", "first_obstacle_collision_time_s",
            "first_any_collision_time_s", "frames", "closest_center_distance_m",
            "visible_through_closest_approach", "verification_passed",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(aggregate, indent=2))
    if not aggregate["all_verification_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
