#!/usr/bin/env python3
"""Render GIFs and summary plots from the saved closed-loop logs."""
from __future__ import annotations

import csv
import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
import numpy as np

from run_experiment import CFG, build_scenario, load_config, make_obstacle
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


RESULTS = HERE / "results"


def load_rows(variant):
    with (RESULTS / variant / "ego_states.csv").open() as f:
        rows = list(csv.DictReader(f))
    numeric = {}
    for key in ["time_s", "rear_axle_x_m", "rear_axle_y_m", "heading_rad", "speed_mps"]:
        numeric[key] = np.asarray([float(row[key]) for row in rows])
    numeric["collision"] = np.asarray([bool(int(row["obstacle_collision"])) for row in rows])
    numeric["collision_latched"] = np.maximum.accumulate(numeric["collision"])
    return rows, numeric


def local_transform(origin, heading):
    c, s = math.cos(heading), math.sin(heading)
    rotation = np.asarray([[c, -s], [s, c]])

    def convert(points):
        return (np.asarray(points, dtype=float) - origin) @ rotation

    return convert


def map_geometry(scenario, world_points, convert):
    center = (world_points.min(axis=0) + world_points.max(axis=0)) / 2
    radius = float(np.linalg.norm(world_points.max(axis=0) - world_points.min(axis=0)) / 2 + 70)
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    objects = scenario.map_api.get_proximal_map_objects(Point2D(*center), radius, layers)
    roads, centerlines = [], []
    seen = set()
    for layer in layers:
        for obj in objects[layer]:
            if obj.id in seen:
                continue
            seen.add(obj.id)
            roads.append(convert(obj.polygon.exterior.coords))
            centerlines.append(convert([[point.x, point.y] for point in obj.baseline_path.discrete_path]))
    return roads, centerlines


def render_variant(variant, scenario, roads, centerlines, convert, origin_heading):
    rows, data = load_rows(variant)
    reference = np.load(RESULTS / variant / "reference_trajectories_global.npz")["trajectories"]
    actual = convert(np.column_stack([data["rear_axle_x_m"], data["rear_axle_y_m"]]))
    expert_states = list(scenario.get_expert_ego_trajectory())
    expert = convert([[state.rear_axle.x, state.rear_axle.y] for state in expert_states])
    obstacle = make_obstacle(scenario.initial_ego_state, CFG["obstacle"], scenario.start_time.time_us)
    obstacle_polygon = convert(obstacle.box.geometry.exterior.coords)
    enabled = variant == "obstacle"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.subplots_adjust(left=.09, right=.98, bottom=.18, top=.85)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#e6eee5")
    ax.add_collection(PolyCollection(roads, facecolors="#fbfbfb", edgecolors="#aab1b7", linewidths=.7, zorder=1))
    ax.add_collection(LineCollection(centerlines, colors="#bbc3c9", linewidths=.6, linestyles="dashed", zorder=2))
    ax.plot(expert[:, 0], expert[:, 1], color="#6b7280", lw=1.4, ls="--", zorder=3)
    if enabled:
        ax.add_collection(PolyCollection([obstacle_polygon], facecolors="#dc2626", edgecolors="#7f1d1d", linewidths=1.5, zorder=7))
    agents = PolyCollection([], linewidths=.5, edgecolors="#6b4b25", zorder=5)
    ego = PolyCollection([], facecolors="#2563eb", edgecolors="#173b78", zorder=8)
    ax.add_collection(agents)
    ax.add_collection(ego)
    driven, = ax.plot([], [], color="#2563eb", lw=2.5, zorder=6)
    planned, = ax.plot([], [], color="#00a6ad", lw=2.5, zorder=6)
    status = ax.text(.02, .97, "", transform=ax.transAxes, va="top", fontsize=10.5,
                     bbox=dict(facecolor="white", alpha=.93, edgecolor="none"), zorder=10)
    fig.suptitle(
        f"Flow Planner real closed loop | {variant}\n"
        "nuPlan LQR + kinematic bicycle | non-reactive log replay",
        fontsize=13,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forward distance from initial pose (m)")
    ax.set_ylabel("Lateral distance (m)")
    legend_items = [
        Line2D([], [], color="#00a6ad", lw=3, label="Current Flow Planner reference"),
        Line2D([], [], color="#2563eb", lw=3, label="Executed actual path"),
        Line2D([], [], color="#6b7280", lw=2, ls="--", label="Logged human path"),
        Line2D([], [], color="#ed9b40", lw=6, label="Replayed traffic"),
    ]
    if enabled:
        legend_items.append(
            Line2D([], [], color="#dc2626", lw=7, label=f"Added {CFG['obstacle'].get('display_name', 'obstacle')}")
        )
    fig.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3, frameon=False, fontsize=8.5)
    vehicle = scenario.ego_vehicle_parameters

    def update(index):
        observation = scenario.get_tracked_objects_at_iteration(index)
        polygons, colors = [], []
        for obj in observation.tracked_objects.tracked_objects:
            polygons.append(convert(obj.box.geometry.exterior.coords))
            colors.append("#ed9b40" if obj.tracked_object_type.name == "VEHICLE" else "#a855b5")
        agents.set_verts(polygons)
        agents.set_facecolors(colors)
        footprint = CarFootprint.build_from_rear_axle(
            StateSE2(data["rear_axle_x_m"][index], data["rear_axle_y_m"][index], data["heading_rad"][index]), vehicle
        )
        ego.set_verts([convert(footprint.geometry.exterior.coords)])
        driven.set_data(actual[: index + 1, 0], actual[: index + 1, 1])
        future = convert(reference[index, :, 1:3])
        planned.set_data(future[:, 0], future[:, 1])
        x, y = actual[index]
        ax.set_xlim(x - 25, x + 65)
        ax.set_ylim(y - 24, y + 24)
        collision = bool(data["collision_latched"][index])
        status.set_text(
            f"t = {data['time_s'][index]:4.1f} s    speed = {data['speed_mps'][index]:4.1f} m/s\n"
            f"obstacle collision so far: {'YES' if collision else 'no'}"
        )
        status.set_color("#b91c1c" if collision else "#111827")
        return agents, ego, driven, planned, status

    indices = list(range(0, len(rows), 2))
    animation = FuncAnimation(fig, update, frames=indices, interval=200, blit=False)
    output = RESULTS / f"{variant}_animation.gif"
    animation.save(output, writer=PillowWriter(fps=5))
    update(indices[len(indices) // 2])
    fig.savefig(RESULTS / f"{variant}_preview.png", dpi=140)
    plt.close(fig)
    return output


def comparison_plots(scenario, convert):
    _, baseline = load_rows("baseline")
    _, obstacle = load_rows("obstacle")
    bxy = convert(np.column_stack([baseline["rear_axle_x_m"], baseline["rear_axle_y_m"]]))
    oxy = convert(np.column_stack([obstacle["rear_axle_x_m"], obstacle["rear_axle_y_m"]]))
    fixed = make_obstacle(scenario.initial_ego_state, CFG["obstacle"], scenario.start_time.time_us)
    obstacle_polygon = convert(fixed.box.geometry.exterior.coords)

    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150)
    ax.plot(bxy[:, 0], bxy[:, 1], color="#2563eb", lw=2.3, label="Baseline actual")
    ax.plot(oxy[:, 0], oxy[:, 1], color="#f59e0b", lw=2.3, label="Stopped-vehicle actual")
    ax.fill(
        obstacle_polygon[:, 0],
        obstacle_polygon[:, 1],
        color="#dc2626",
        alpha=.75,
        label=f"Added {CFG['obstacle'].get('display_name', 'obstacle')}",
    )
    ax.set_aspect("equal", adjustable="box")
    all_x = np.concatenate([bxy[:, 0], oxy[:, 0], obstacle_polygon[:, 0]])
    all_y = np.concatenate([bxy[:, 1], oxy[:, 1], obstacle_polygon[:, 1]])
    ax.set_xlim(all_x.min() - 3, all_x.max() + 3)
    ax.set_ylim(min(-4.0, all_y.min() - 2), max(4.0, all_y.max() + 2))
    ax.grid(alpha=.25)
    ax.set_xlabel("Forward distance from initial pose (m)")
    ax.set_ylabel("Lateral distance (m)")
    ax.set_title("Executed trajectory comparison (real closed-loop logs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "actual_trajectory_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.plot(baseline["time_s"], baseline["speed_mps"], color="#2563eb", lw=2.2, label="Baseline")
    ax.plot(obstacle["time_s"], obstacle["speed_mps"], color="#f59e0b", lw=2.2, label="Stopped vehicle")
    collision_indices = np.flatnonzero(obstacle["collision"])
    if len(collision_indices):
        first = collision_indices[0]
        ax.axvline(obstacle["time_s"][first], color="#dc2626", ls="--", lw=1.7, label="First obstacle collision")
    ax.grid(alpha=.25)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Actual ego speed (m/s)")
    ax.set_title("Speed over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "speed_time.png")
    plt.close(fig)


def main():
    global CFG, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = parser.parse_args()
    CFG = load_config(args.config.resolve())
    RESULTS = Path(CFG["project"].get("results_dir", HERE / "results"))
    for variant in ("baseline", "obstacle"):
        if not (RESULTS / variant / "summary.json").is_file():
            raise RuntimeError(f"Missing completed result: {variant}")
    scenario, _ = build_scenario(CFG)
    _, baseline = load_rows("baseline")
    world = np.column_stack([baseline["rear_axle_x_m"], baseline["rear_axle_y_m"]])
    origin = world[0]
    heading = baseline["heading_rad"][0]
    convert = local_transform(origin, heading)
    roads, centerlines = map_geometry(scenario, world, convert)
    for variant in ("baseline", "obstacle"):
        path = render_variant(variant, scenario, roads, centerlines, convert, heading)
        print(path)
    comparison_plots(scenario, convert)


if __name__ == "__main__":
    main()

