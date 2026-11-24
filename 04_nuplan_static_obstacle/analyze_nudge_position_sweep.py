#!/usr/bin/env python3
"""Analyze and plot the controlled nudge stopped-vehicle position sweep."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import run_official_interplan  # Sets the local Flow Planner / nuPlan import paths.
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results_nudge_position_sweep"
CONFIG = json.loads((HERE / "nudge_position_sweep.json").read_text())
COLORS = {
    "baseline": "#111827",
    "closer_8m": "#dc2626",
    "farther_8m": "#2563eb",
    "left_1m": "#16a34a",
    "right_1m": "#f59e0b",
}
LABELS = {
    "baseline": "Baseline",
    "closer_8m": "8 m closer",
    "farther_8m": "8 m farther",
    "left_1m": "1 m left",
    "right_1m": "1 m right",
}


def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def local_xy(points, origin, heading):
    points = np.asarray(points, dtype=float) - origin
    c, s = math.cos(heading), math.sin(heading)
    return np.column_stack((c * points[:, 0] + s * points[:, 1], -s * points[:, 0] + c * points[:, 1]))


def main():
    variants = [item["id"] for item in CONFIG["variants"]]
    data = {}
    for variant in variants:
        summary = json.loads((ROOT / variant / "summary.json").read_text())
        rows = load_csv(ROOT / variant / "ego_states.csv")
        traffic = json.loads((ROOT / variant / "traffic_tracks.json").read_text())
        world = np.asarray([[float(r["rear_axle_x_m"]), float(r["rear_axle_y_m"])] for r in rows])
        data[variant] = {"summary": summary, "rows": rows, "world": world, "traffic": traffic}
    origin = data["baseline"]["world"][0]
    heading = float(data["baseline"]["rows"][0]["heading_rad"])
    baseline_local = local_xy(data["baseline"]["world"], origin, heading)
    metrics = []
    verification = {}
    for variant in variants:
        item = data[variant]
        summary, rows = item["summary"], item["rows"]
        local = local_xy(item["world"], origin, heading)
        obstacle_pose = summary["stopped_vehicle_poses"][0]
        obstacle_local = local_xy([obstacle_pose[:2]], origin, heading)[0]
        distances = np.linalg.norm(item["world"] - np.asarray(obstacle_pose[:2]), axis=1)
        closest = int(np.argmin(distances))
        station = int(np.argmin(np.abs(local[:, 0] - obstacle_local[0])))
        common_min = max(float(local[:, 0].min()), float(baseline_local[:, 0].min()))
        common_max = min(float(local[:, 0].max()), float(baseline_local[:, 0].max()))
        grid = np.linspace(common_min, common_max, 500)
        variant_lat = np.interp(grid, local[:, 0], local[:, 1])
        baseline_lat = np.interp(grid, baseline_local[:, 0], baseline_local[:, 1])
        window = np.abs(grid - obstacle_local[0]) <= 15.0
        collision_time = summary["first_any_collision_time_s"]
        times = np.asarray([float(row["time_s"]) for row in rows])
        speeds = np.asarray([float(row["speed_mps"]) for row in rows])
        references = np.load(ROOT / variant / "reference_trajectories_global.npz")["trajectories"]
        controls = load_csv(ROOT / variant / "lqr_controls.csv")
        audits = json.loads((ROOT / variant / "model_input_audit.json").read_text())
        tracks = np.load(ROOT / variant / "stopped_vehicle_tracks.npz")["vehicle_0"]
        pose_error = np.linalg.norm(tracks[:, 1:3] - np.asarray(obstacle_pose[:2]), axis=1)
        checks = {
            "complete_state_log": len(rows) == summary["frames"],
            "complete_reference_log": references.shape == (summary["planner_calls"], 81, 5),
            "complete_control_log": len(controls) == len(rows) - 1,
            "obstacle_entered_every_model_input": len(audits) == summary["planner_calls"] and all(
                audit["stopped_vehicles"][0]["slot"] is not None for audit in audits
            ),
            "obstacle_static_at_configured_pose": float(np.max(pose_error)) < 1e-6 and float(np.max(tracks[:, 4])) < 1e-6,
        }
        verification[variant] = {"passed": all(checks.values()), "checks": checks}
        metrics.append({
            "variant": variant,
            "longitudinal_shift_m": summary["sweep_longitudinal_m"],
            "lateral_shift_m": summary["sweep_lateral_m"],
            "initial_obstacle_distance_m": float(np.linalg.norm(item["world"][0] - obstacle_pose[:2])),
            "minimum_speed_mps": summary["minimum_speed_mps"],
            "time_of_minimum_speed_s": float(times[int(np.argmin(speeds))]),
            "final_speed_mps": summary["final_speed_mps"],
            "collision": collision_time is not None,
            "first_collision_time_s": collision_time,
            "speed_at_first_collision_mps": None if collision_time is None else float(np.interp(collision_time, times, speeds)),
            "closest_center_distance_m": float(distances[closest]),
            "ego_lateral_at_obstacle_station_m": float(local[station, 1]),
            "obstacle_lateral_m": float(obstacle_local[1]),
            "center_lateral_separation_at_station_m": float(abs(local[station, 1] - obstacle_local[1])),
            "maximum_path_change_vs_baseline_near_obstacle_m": float(np.max(np.abs(variant_lat[window] - baseline_lat[window]))),
            "obstacle_visible_calls": summary["stopped_vehicles_seen_calls"],
            "planner_calls": summary["planner_calls"],
        })
        item["local"] = local
        item["obstacle_local"] = obstacle_local

    with (ROOT / "sweep_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (ROOT / "sweep_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    for variant in variants:
        item = data[variant]
        ax.plot(item["local"][:, 0], item["local"][:, 1], color=COLORS[variant], lw=2.4, label=LABELS[variant])
        pose = item["summary"]["stopped_vehicle_poses"][0]
        tracked = min(
            item["traffic"][0]["objects"],
            key=lambda obj: math.hypot(obj["x"] - pose[0], obj["y"] - pose[1]),
        )
        box = OrientedBox(StateSE2(*pose), tracked["length"], tracked["width"], tracked["height"])
        poly = local_xy(np.asarray(box.geometry.exterior.coords), origin, heading)
        ax.add_patch(Polygon(poly, closed=True, facecolor=COLORS[variant], edgecolor=COLORS[variant], alpha=.18))
    ax.set_xlim(-3, 75)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forward from initial ego pose (m)")
    ax.set_ylabel("Lateral from initial ego pose (m)")
    ax.set_title("Executed trajectories after shifting the interPlan stopped vehicle")
    ax.grid(alpha=.22)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "trajectory_position_sweep.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    for variant in variants:
        rows = data[variant]["rows"]
        times = [float(row["time_s"]) for row in rows]
        speeds = [float(row["speed_mps"]) for row in rows]
        ax.plot(times, speeds, color=COLORS[variant], lw=2.2, label=LABELS[variant])
        collision = data[variant]["summary"]["first_any_collision_time_s"]
        if collision is not None:
            ax.scatter([collision], [np.interp(collision, times, speeds)], color=COLORS[variant], marker="x", s=70, zorder=5)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Actual ego speed (m/s)")
    ax.set_title("Speed response to stopped-vehicle position")
    ax.grid(alpha=.22)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "speed_position_sweep.png")
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
