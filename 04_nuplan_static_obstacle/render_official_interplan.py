#!/usr/bin/env python3
"""Render an official interPlan run from its saved closed-loop logs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
import numpy as np

from run_official_interplan import RESULTS, build_scenario
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def transform(origin, heading):
    c, s = math.cos(heading), math.sin(heading)
    rotation = np.asarray([[c, -s], [s, c]])
    return lambda points: (np.asarray(points, dtype=float) - origin) @ rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token")
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    args = parser.parse_args()
    directory = args.results_root / args.token
    if not (directory / "traffic_tracks.json").is_file():
        raise RuntimeError(f"{args.token} needs a forced rerun with traffic logging")
    summary = json.loads((directory / "summary.json").read_text())
    scenario, _ = build_scenario(summary.get("modified_token", args.token))
    with (directory / "ego_states.csv").open() as f:
        rows = list(csv.DictReader(f))
    actual_world = np.asarray([[float(row["rear_axle_x_m"]), float(row["rear_axle_y_m"])] for row in rows])
    headings = np.asarray([float(row["heading_rad"]) for row in rows])
    speeds = np.asarray([float(row["speed_mps"]) for row in rows])
    times = np.asarray([float(row["time_s"]) for row in rows])
    collision = np.maximum.accumulate([int(row["any_collision"]) for row in rows]).astype(bool)
    references = np.load(directory / "reference_trajectories_global.npz")["trajectories"]
    traffic = json.loads((directory / "traffic_tracks.json").read_text())
    stopped_poses = summary["stopped_vehicle_poses"]
    origin, initial_heading = actual_world[0], headings[0]
    convert = transform(origin, initial_heading)
    actual = convert(actual_world)
    expert = convert([[state.rear_axle.x, state.rear_axle.y] for state in scenario.get_expert_ego_trajectory()])

    center = (actual_world.min(axis=0) + actual_world.max(axis=0)) / 2
    radius = float(np.linalg.norm(actual_world.max(axis=0) - actual_world.min(axis=0)) / 2 + 70)
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    nearby = scenario.map_api.get_proximal_map_objects(Point2D(*center), radius, layers)
    roads, centerlines = [], []
    seen = set()
    for layer in layers:
        for obj in nearby[layer]:
            if obj.id in seen:
                continue
            seen.add(obj.id)
            roads.append(convert(obj.polygon.exterior.coords))
            centerlines.append(convert([[p.x, p.y] for p in obj.baseline_path.discrete_path]))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.subplots_adjust(left=.09, right=.98, bottom=.18, top=.86)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#e6eee5")
    ax.add_collection(PolyCollection(roads, facecolors="#fbfbfb", edgecolors="#aab1b7", linewidths=.7, zorder=1))
    ax.add_collection(LineCollection(centerlines, colors="#bbc3c9", linewidths=.6, linestyles="dashed", zorder=2))
    ax.plot(expert[:, 0], expert[:, 1], color="#6b7280", lw=1.4, ls="--", zorder=3)
    agents = PolyCollection([], linewidths=.6, edgecolors="#6b4b25", zorder=5)
    ego = PolyCollection([], facecolors="#2563eb", edgecolors="#173b78", zorder=8)
    ax.add_collection(agents)
    ax.add_collection(ego)
    driven, = ax.plot([], [], color="#2563eb", lw=2.6, zorder=6)
    planned, = ax.plot([], [], color="#00a6ad", lw=2.6, zorder=6)
    status = ax.text(.02, .97, "", transform=ax.transAxes, va="top", fontsize=10.5,
                     bbox=dict(facecolor="white", alpha=.93, edgecolor="none"), zorder=10)
    fig.suptitle(f"Official interPlan + Flow Planner | {args.token}\n{scenario.scenario_type}", fontsize=13)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forward distance from initial pose (m)")
    ax.set_ylabel("Lateral distance (m)")
    fig.legend(
        handles=[
            Line2D([], [], color="#00a6ad", lw=3, label="Current Flow Planner reference"),
            Line2D([], [], color="#2563eb", lw=3, label="Executed actual path"),
            Line2D([], [], color="#6b7280", lw=2, ls="--", label="Logged expert path"),
            Line2D([], [], color="#dc2626", lw=7, label="interPlan stopped vehicle"),
            Line2D([], [], color="#ed9b40", lw=7, label="Reactive traffic"),
        ],
        loc="lower center", bbox_to_anchor=(.5, .035), ncol=3, frameon=False, fontsize=8.5,
    )
    vehicle = scenario.ego_vehicle_parameters

    def update(index):
        polygons, colors = [], []
        for obj in traffic[index]["objects"]:
            pose = StateSE2(obj["x"], obj["y"], obj["heading"])
            from nuplan.common.actor_state.oriented_box import OrientedBox
            box = OrientedBox(pose, obj["length"], obj["width"], obj["height"])
            polygons.append(convert(box.geometry.exterior.coords))
            is_stopped = any(math.hypot(obj["x"] - p[0], obj["y"] - p[1]) < .3 for p in stopped_poses)
            colors.append("#dc2626" if is_stopped else ("#ed9b40" if obj["type"] == "VEHICLE" else "#a855b5"))
        agents.set_verts(polygons)
        agents.set_facecolors(colors)
        footprint = CarFootprint.build_from_rear_axle(
            StateSE2(actual_world[index, 0], actual_world[index, 1], headings[index]), vehicle
        )
        ego.set_verts([convert(footprint.geometry.exterior.coords)])
        driven.set_data(actual[:index + 1, 0], actual[:index + 1, 1])
        future = convert(references[index, :, 1:3])
        planned.set_data(future[:, 0], future[:, 1])
        x, y = actual[index]
        ax.set_xlim(x - 25, x + 65)
        ax.set_ylim(y - 24, y + 24)
        status.set_text(
            f"t = {times[index]:4.1f} s    speed = {speeds[index]:4.1f} m/s\n"
            f"collision so far: {'YES' if collision[index] else 'no'}"
        )
        status.set_color("#b91c1c" if collision[index] else "#111827")
        return agents, ego, driven, planned, status

    indices = list(range(0, len(rows), 2))
    animation = FuncAnimation(fig, update, frames=indices, interval=200, blit=False)
    output = args.results_root / f"{args.token}_animation.gif"
    animation.save(output, writer=PillowWriter(fps=5))
    update(indices[len(indices) // 2])
    fig.savefig(args.results_root / f"{args.token}_preview.png", dpi=140)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
