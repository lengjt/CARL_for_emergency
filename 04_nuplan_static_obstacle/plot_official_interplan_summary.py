#!/usr/bin/env python3
"""Create compact summary plots from official interPlan closed-loop logs."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_official_interplan import RESULTS
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2

REPRESENTATIVES = [
    ("81f400b414215325-s0", "Nudge: moving avoidance", "#16a34a"),
    ("168c4e592f1b5984-s1", "Overtake: moving avoidance", "#2563eb"),
    ("948201ecfe2c560a-s1", "Crashed cars: safe stop", "#7c3aed"),
    ("9e7581ef72155c9f-s0", "Overtake: collision", "#dc2626"),
]


def rows(token):
    with (RESULTS / token / "ego_states.csv").open() as f:
        return list(csv.DictReader(f))


def main():
    aggregate = json.loads((RESULTS / "aggregate_summary.json").read_text())
    categories = list(aggregate["categories"])
    labels = ["Crashed cars", "Nudge", "Overtake"]
    modes = ["moving_avoidance", "slow_or_stop_then_go", "safe_stop", "collision"]
    colors = ["#16a34a", "#f59e0b", "#7c3aed", "#dc2626"]
    mode_labels = ["Moving avoidance", "Slow/stop then go", "Safe stop", "Collision"]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    bottom = np.zeros(len(categories))
    for mode, color, label in zip(modes, colors, mode_labels):
        values = np.asarray([aggregate["categories"][cat]["behavior_counts"].get(mode, 0) for cat in categories])
        ax.bar(labels, values, bottom=bottom, color=color, label=label)
        bottom += values
    ax.set_ylim(0, 10.7)
    ax.set_ylabel("Official scenarios")
    ax.set_title("Released Flow Planner on official interPlan scenarios")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTS / "category_outcomes.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for token, label, color in REPRESENTATIVES:
        data = rows(token)
        ax.plot(
            [float(row["time_s"]) for row in data],
            [float(row["speed_mps"]) for row in data],
            color=color,
            lw=2.1,
            label=label,
        )
        collision = next((float(row["time_s"]) for row in data if int(row["any_collision"])), None)
        if collision is not None:
            ax.axvline(collision, color=color, ls="--", lw=1.2, alpha=.8)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Actual ego speed (m/s)")
    ax.set_title("Representative official interPlan speed profiles")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "representative_speed_profiles.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    for ax, (token, label, color) in zip(axes.flat, REPRESENTATIVES):
        data = rows(token)
        x = np.asarray([float(row["rear_axle_x_m"]) for row in data])
        y = np.asarray([float(row["rear_axle_y_m"]) for row in data])
        h = float(data[0]["heading_rad"])
        c, s = math.cos(h), math.sin(h)
        dx, dy = x - x[0], y - y[0]
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        summary = json.loads((RESULTS / token / "summary.json").read_text())
        ax.plot(local_x, local_y, color=color, lw=2.4, label="Executed path")
        traffic = json.loads((RESULTS / token / "traffic_tracks.json").read_text())[0]["objects"]
        for pose in summary["stopped_vehicle_poses"]:
            obj = min(traffic, key=lambda item: math.hypot(item["x"] - pose[0], item["y"] - pose[1]))
            box = OrientedBox(StateSE2(pose[0], pose[1], pose[2]), obj["length"], obj["width"], obj["height"])
            polygon = np.asarray(box.geometry.exterior.coords)
            pdx, pdy = polygon[:, 0] - x[0], polygon[:, 1] - y[0]
            ax.fill(c * pdx + s * pdy, -s * pdx + c * pdy, color="#dc2626", alpha=.75)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(label)
        ax.set_xlabel("Forward (m)")
        ax.set_ylabel("Lateral (m)")
        ax.grid(alpha=.2)
    fig.suptitle("Executed trajectories and official interPlan obstacles")
    fig.tight_layout()
    fig.savefig(RESULTS / "representative_trajectories.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
