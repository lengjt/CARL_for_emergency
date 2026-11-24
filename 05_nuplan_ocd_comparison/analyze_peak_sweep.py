#!/usr/bin/env python3
"""Verify and compare the low/medium/high OCD MF5.2 peak-scale nudge runs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_comparison import local_xy, signed_one_step_tracking_error

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results_ocd_peak_sweep"
CONFIG = json.loads((HERE / "ocd_peak_sweep.json").read_text())
COLORS = {"low_0p5": "#dc2626", "medium_0p8": "#f59e0b", "high_1p0": "#2563eb"}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    levels = CONFIG["levels"]
    data, verification, metrics = {}, {}, {}
    for level in levels:
        result_id = level["id"]
        expected_scale = float(level["internal_peak_scale"])
        directory = ROOT / result_id
        summary = json.loads((directory / "summary.json").read_text())
        rows = read_csv(directory / "ego_states.csv")
        controls = read_csv(directory / "lqr_controls.csv")
        ocd = read_csv(directory / "ocd_states_controls.csv")
        audits = json.loads((directory / "model_input_audit.json").read_text())
        references = np.load(directory / "reference_trajectories_global.npz")["trajectories"]
        tracks = np.load(directory / "stopped_vehicle_tracks.npz")["vehicle_0"]
        pose = np.asarray(summary["stopped_vehicle_poses"][0][:2])
        scale_keys = ["ocd_peak_scale_fl", "ocd_peak_scale_fr", "ocd_peak_scale_rl", "ocd_peak_scale_rr"]
        checks = {
            "complete_fixed_horizon": len(rows) == 181 and references.shape == (181, 81, 5),
            "complete_control_and_ocd_logs": len(controls) == len(ocd) == 180,
            "obstacle_visible_every_call": len(audits) == 181 and all(
                audit["stopped_vehicles"][0]["slot"] is not None for audit in audits
            ),
            "obstacle_static": float(np.max(np.linalg.norm(tracks[:, 1:3] - pose, axis=1))) < 1e-6
            and float(np.max(tracks[:, 4])) < 1e-6,
            "configured_scale_applied_to_all_wheels_every_step": all(
                all(abs(float(item[key]) - expected_scale) < 1e-12 for key in scale_keys) for item in ocd
            ),
            "closed_loop_feedback_timestamps_match": all(
                int(item["timestamp_us"]) == int(row["timestamp_us"]) for item, row in zip(ocd, rows[1:])
            ),
            "released_checkpoint_and_sampling": summary["checkpoint_sha256"]
            == "9231b4c02968c9e42dd2881ad3e7dd4bd362796f7b2c0158e0458c4ab3bde88e"
            and summary["seed"] == 0
            and summary["sample_steps"] == 4,
        }
        if "initial_speed_mps" in CONFIG:
            checks["requested_initial_speed_applied"] = (
                abs(summary["initial_speed_mps"] - float(CONFIG["initial_speed_mps"])) < 1e-12
                and abs(float(rows[0]["speed_mps"]) - float(CONFIG["initial_speed_mps"])) < 1e-12
                and summary.get("initial_speed_override", {}).get("target_speed_mps")
                == float(CONFIG["initial_speed_mps"])
            )
        verification[result_id] = {"passed": all(checks.values()), "checks": checks}
        world = np.asarray([[float(row["rear_axle_x_m"]), float(row["rear_axle_y_m"])] for row in rows])
        data[result_id] = {
            "level": level,
            "summary": summary,
            "rows": rows,
            "ocd": ocd,
            "references": references,
            "world": world,
            "tracking_error": signed_one_step_tracking_error(rows, references),
        }

    initial = data[levels[0]["id"]]["rows"][0]
    origin = np.asarray([float(initial["rear_axle_x_m"]), float(initial["rear_axle_y_m"])])
    heading = float(initial["heading_rad"])
    obstacle_world = np.asarray(data[levels[0]["id"]]["summary"]["stopped_vehicle_poses"][0][:2])
    obstacle_local = local_xy([obstacle_world], origin, heading)[0]
    for level in levels:
        result_id = level["id"]
        item = data[result_id]
        local = local_xy(item["world"], origin, heading)
        rows, ocd = item["rows"], item["ocd"]
        times = np.asarray([float(row["time_s"]) for row in rows])
        speeds = np.asarray([float(row["speed_mps"]) for row in rows])
        distance = np.linalg.norm(item["world"] - obstacle_world, axis=1)
        station = int(np.argmin(np.abs(local[:, 0] - obstacle_local[0])))
        sideslip = [math.atan2(float(row["ocd_vy_mps"]), max(abs(float(row["ocd_vx_mps"])), .2)) for row in ocd]
        lateral_acceleration = [float(row["ocd_ay_mps2"]) for row in ocd]
        collision_time = item["summary"]["first_any_collision_time_s"]
        collision_index = None if collision_time is None else int(np.argmin(np.abs(times - collision_time)))
        metrics[result_id] = {
            "internal_peak_scale": level["internal_peak_scale"],
            "collision": collision_time is not None,
            "first_collision_time_s": collision_time,
            "speed_at_first_collision_mps": None if collision_index is None else float(speeds[collision_index]),
            "lateral_position_at_first_collision_m": None if collision_index is None else float(local[collision_index, 1]),
            "center_distance_at_first_collision_m": None if collision_index is None else float(distance[collision_index]),
            "minimum_speed_mps": float(speeds.min()),
            "final_speed_mps": float(speeds[-1]),
            "closest_obstacle_center_distance_m": float(distance.min()),
            "lateral_position_at_obstacle_station_m": float(local[station, 1]),
            "rms_one_step_lateral_tracking_error_m": float(np.sqrt(np.mean(item["tracking_error"][1:] ** 2))),
            "maximum_abs_one_step_lateral_tracking_error_m": float(np.max(np.abs(item["tracking_error"][1:]))),
            "maximum_abs_cg_sideslip_rad": float(np.max(np.abs(sideslip))),
            "maximum_abs_yaw_rate_radps": item["summary"]["maximum_abs_yaw_rate_radps"],
            "maximum_abs_lateral_acceleration_mps2": float(np.max(np.abs(lateral_acceleration))),
        }
        item.update({"local": local, "times": times, "speeds": speeds, "sideslip": np.asarray(sideslip)})

    high = data["high_1p0"]
    comparison = {}
    for result_id in ["low_0p5", "medium_0p8"]:
        current = data[result_id]
        forward_min = max(current["local"][:, 0].min(), high["local"][:, 0].min())
        forward_max = min(current["local"][:, 0].max(), high["local"][:, 0].max())
        grid = np.linspace(forward_min, forward_max, 1000)
        current_y = np.interp(grid, current["local"][:, 0], current["local"][:, 1])
        high_y = np.interp(grid, high["local"][:, 0], high["local"][:, 1])
        comparison[f"{result_id}_vs_high_1p0"] = {
            "maximum_lateral_path_difference_m": float(np.max(np.abs(current_y - high_y))),
            "rms_lateral_path_difference_m": float(np.sqrt(np.mean((current_y - high_y) ** 2))),
            "final_position_difference_m": float(np.linalg.norm(current["world"][-1] - high["world"][-1])),
        }
    output = {"levels": metrics, "comparisons": comparison}
    (ROOT / "peak_sweep_metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=160)
    for level in levels:
        result_id = level["id"]
        ax.plot(data[result_id]["local"][:, 0], data[result_id]["local"][:, 1], color=COLORS[result_id], lw=2.4, label=level["label"])
    ax.scatter([obstacle_local[0]], [obstacle_local[1]], marker="s", s=150, color="#7f1d1d", label="Stopped vehicle center")
    ax.set_xlabel("Forward from initial rear axle (m)")
    ax.set_ylabel("Lateral from initial rear axle (m)")
    ax.set_title("Executed paths under three internal MF5.2 peak scales")
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(ROOT / "peak_sweep_trajectory_overlay.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    for level in levels:
        result_id = level["id"]
        ax.plot(data[result_id]["times"], data[result_id]["speeds"], color=COLORS[result_id], lw=2.2, label=level["label"])
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Actual ego speed (m/s)")
    ax.set_title("Speed under three internal MF5.2 peak scales")
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "peak_sweep_speed.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    for level in levels:
        result_id = level["id"]
        ax.plot(data[result_id]["times"], data[result_id]["tracking_error"], color=COLORS[result_id], lw=2.0, label=level["label"])
    ax.axhline(0, color="#6b7280", lw=.8)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Signed one-step lateral tracking error (m)")
    ax.set_title("Tracking error under three internal MF5.2 peak scales")
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "peak_sweep_tracking_error.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=160, sharex=True)
    for level in levels:
        result_id = level["id"]
        ocd = data[result_id]["ocd"]
        control_times = data[result_id]["times"][1:]
        axes[0].plot(control_times, data[result_id]["sideslip"], color=COLORS[result_id], lw=2, label=level["label"])
        axes[1].plot(control_times, [float(row["ocd_ay_mps2"]) for row in ocd], color=COLORS[result_id], lw=2, label=level["label"])
    axes[0].set_ylabel("CG sideslip angle (rad)")
    axes[1].set_ylabel("OCD lateral acceleration (m/s²)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[0].set_title("OCD dynamic response")
    for ax in axes:
        ax.grid(alpha=.22)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "peak_sweep_dynamic_response.png")
    plt.close(fig)
    print(json.dumps({"verification": verification, "metrics": output}, indent=2))


if __name__ == "__main__":
    main()
