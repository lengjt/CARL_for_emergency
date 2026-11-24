#!/usr/bin/env python3
"""Verify and compare the kinematic and OCD closed-loop results."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
STAGE1 = RESEARCH / "04_nuplan_static_obstacle"
sys.path.insert(0, str(STAGE1))
from run_official_interplan import build_scenario  # noqa: E402

ROOT = HERE / "results"
MODES = ["kinematic", "ocd"]
COLORS = {"kinematic": "#2563eb", "ocd": "#dc2626"}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def local_xy(points, origin, heading):
    points = np.asarray(points, dtype=float) - origin
    c, s = math.cos(heading), math.sin(heading)
    return np.column_stack((c * points[:, 0] + s * points[:, 1], -s * points[:, 0] + c * points[:, 1]))


def signed_one_step_tracking_error(rows, references):
    errors = [0.0]
    for index in range(1, len(rows)):
        timestamp = int(rows[index]["timestamp_us"])
        previous_plan = references[index - 1]
        point_index = int(np.argmin(np.abs(previous_plan[:, 0] - timestamp)))
        reference = previous_plan[point_index]
        dx = float(rows[index]["rear_axle_x_m"]) - reference[1]
        dy = float(rows[index]["rear_axle_y_m"]) - reference[2]
        errors.append(-math.sin(reference[3]) * dx + math.cos(reference[3]) * dy)
    return np.asarray(errors)


def main():
    data = {}
    verification = {}
    for mode in MODES:
        directory = ROOT / mode
        summary = json.loads((directory / "summary.json").read_text())
        rows = read_csv(directory / "ego_states.csv")
        controls = read_csv(directory / "lqr_controls.csv")
        audits = json.loads((directory / "model_input_audit.json").read_text())
        references = np.load(directory / "reference_trajectories_global.npz")["trajectories"]
        tracks = np.load(directory / "stopped_vehicle_tracks.npz")["vehicle_0"]
        pose = np.asarray(summary["stopped_vehicle_poses"][0][:2])
        pose_error = np.linalg.norm(tracks[:, 1:3] - pose, axis=1)
        checks = {
            "fixed_horizon_state_log": len(rows) == summary["fixed_horizon_steps"] == 181,
            "complete_reference_log": references.shape == (181, 81, 5),
            "complete_control_log": len(controls) == 180,
            "obstacle_visible_every_planning_call": len(audits) == 181 and all(
                audit["stopped_vehicles"][0]["slot"] is not None for audit in audits
            ),
            "obstacle_static_and_geometry_consistent": float(np.max(pose_error)) < 1e-6
            and float(np.max(tracks[:, 4])) < 1e-6,
            "released_checkpoint_and_sampling": summary["checkpoint_sha256"]
            == "9231b4c02968c9e42dd2881ad3e7dd4bd362796f7b2c0158e0458c4ab3bde88e"
            and summary["seed"] == 0
            and summary["cfg_weight"] == 1.0
            and summary["sample_steps"] == 4,
        }
        if mode == "ocd":
            ocd = read_csv(directory / "ocd_states_controls.csv")
            next_rows = rows[1:]
            checks.update(
                {
                    "ocd_updated_every_control_step": len(ocd) == len(next_rows) == 180,
                    "ocd_timestamps_feed_back_to_nuplan": all(
                        int(a["timestamp_us"]) == int(b["timestamp_us"]) for a, b in zip(ocd, next_rows)
                    ),
                    "ocd_yaw_rate_feedback_matches": max(
                        abs(float(a["ocd_yaw_rate_radps"]) - float(b["yaw_rate_radps"]))
                        for a, b in zip(ocd, next_rows)
                    )
                    < 1e-10,
                    "ocd_uses_100_substeps": all(int(item["physics_substeps"]) == 100 for item in ocd),
                    "ocd_regular_peak_scale": all(
                        all(abs(float(item[key]) - 1.0) < 1e-12 for key in (
                            "ocd_peak_scale_fl", "ocd_peak_scale_fr", "ocd_peak_scale_rl", "ocd_peak_scale_rr"
                        ))
                        for item in ocd
                    ),
                    "time_quantization_below_half_physics_step": max(
                        abs(float(item["time_quantization_error_s"])) for item in ocd
                    )
                    < 0.0005,
                }
            )
        verification[mode] = {"passed": all(checks.values()), "checks": checks}
        data[mode] = {
            "summary": summary,
            "rows": rows,
            "controls": controls,
            "references": references,
            "tracking_error": signed_one_step_tracking_error(rows, references),
        }

    initial = data["kinematic"]["rows"][0]
    origin = np.asarray([float(initial["rear_axle_x_m"]), float(initial["rear_axle_y_m"])])
    heading = float(initial["heading_rad"])
    obstacle_world = data["kinematic"]["summary"]["stopped_vehicle_poses"][0][:2]
    obstacle_local = local_xy([obstacle_world], origin, heading)[0]
    metrics = {}
    for mode in MODES:
        rows = data[mode]["rows"]
        world = np.asarray([[float(row["rear_axle_x_m"]), float(row["rear_axle_y_m"])] for row in rows])
        local = local_xy(world, origin, heading)
        times = np.asarray([float(row["time_s"]) for row in rows])
        speeds = np.asarray([float(row["speed_mps"]) for row in rows])
        distance = np.linalg.norm(world - np.asarray(obstacle_world), axis=1)
        station = int(np.argmin(np.abs(local[:, 0] - obstacle_local[0])))
        error = data[mode]["tracking_error"]
        metrics[mode] = {
            "collision": data[mode]["summary"]["first_any_collision_time_s"] is not None,
            "first_collision_time_s": data[mode]["summary"]["first_any_collision_time_s"],
            "minimum_speed_mps": float(speeds.min()),
            "final_speed_mps": float(speeds[-1]),
            "closest_obstacle_center_distance_m": float(distance.min()),
            "lateral_position_at_obstacle_station_m": float(local[station, 1]),
            "rms_signed_one_step_lateral_tracking_error_m": float(np.sqrt(np.mean(error[1:] ** 2))),
            "maximum_abs_one_step_lateral_tracking_error_m": float(np.max(np.abs(error[1:]))),
            "maximum_abs_lateral_velocity_mps": data[mode]["summary"]["maximum_abs_lateral_velocity_mps"],
            "maximum_abs_yaw_rate_radps": data[mode]["summary"]["maximum_abs_yaw_rate_radps"],
        }
        data[mode].update({"world": world, "local": local, "times": times, "speeds": speeds})

    common_forward_min = max(data[m]["local"][:, 0].min() for m in MODES)
    common_forward_max = min(data[m]["local"][:, 0].max() for m in MODES)
    grid = np.linspace(common_forward_min, common_forward_max, 1000)
    lateral = {
        mode: np.interp(grid, data[mode]["local"][:, 0], data[mode]["local"][:, 1]) for mode in MODES
    }
    metrics["comparison"] = {
        "maximum_actual_path_lateral_difference_m": float(np.max(np.abs(lateral["ocd"] - lateral["kinematic"]))),
        "rms_actual_path_lateral_difference_m": float(np.sqrt(np.mean((lateral["ocd"] - lateral["kinematic"]) ** 2))),
        "final_position_difference_m": float(np.linalg.norm(data["ocd"]["world"][-1] - data["kinematic"]["world"][-1])),
    }
    (ROOT / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=160)
    for mode in MODES:
        ax.plot(data[mode]["local"][:, 0], data[mode]["local"][:, 1], color=COLORS[mode], lw=2.5, label=mode.capitalize())
    ax.scatter([obstacle_local[0]], [obstacle_local[1]], marker="s", s=150, color="#991b1b", label="Stopped vehicle center", zorder=5)
    ax.set_xlabel("Forward from initial rear axle (m)")
    ax.set_ylabel("Lateral from initial rear axle (m)")
    ax.set_title("Executed path: kinematic bicycle vs Open Car Dynamics")
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(ROOT / "actual_trajectory_overlay.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    for mode in MODES:
        ax.plot(data[mode]["times"], data[mode]["speeds"], color=COLORS[mode], lw=2.3, label=mode.capitalize())
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Actual ego speed (m/s)")
    ax.set_title("Speed comparison")
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "speed_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    for mode in MODES:
        ax.plot(data[mode]["times"], data[mode]["tracking_error"], color=COLORS[mode], lw=2.0, label=mode.capitalize())
    ax.axhline(0, color="#6b7280", lw=.8)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Signed lateral error to previous cycle's 0.1 s reference (m)")
    ax.set_title("One-step lateral tracking error")
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "lateral_tracking_error.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=160, sharex=True)
    for mode in MODES:
        rows = data[mode]["rows"]
        axes[0].plot(data[mode]["times"], [float(row["rear_axle_vy_mps"]) for row in rows], color=COLORS[mode], lw=2, label=mode.capitalize())
        axes[1].plot(data[mode]["times"], [float(row["yaw_rate_radps"]) for row in rows], color=COLORS[mode], lw=2, label=mode.capitalize())
    axes[0].set_ylabel("Rear-axle lateral velocity (m/s)")
    axes[1].set_ylabel("Yaw rate (rad/s)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[0].set_title("Dynamic states returned to nuPlan")
    for ax in axes:
        ax.grid(alpha=.22)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "dynamic_states_comparison.png")
    plt.close(fig)
    print(json.dumps({"verification": verification, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
