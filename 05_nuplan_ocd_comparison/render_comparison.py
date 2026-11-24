#!/usr/bin/env python3
"""Render synchronized side-by-side kinematic/OCD comparison from saved logs."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
STAGE1 = RESEARCH / "04_nuplan_static_obstacle"
sys.path.insert(0, str(STAGE1))
from run_official_interplan import build_scenario  # noqa: E402
from nuplan.common.actor_state.car_footprint import CarFootprint  # noqa: E402
from nuplan.common.actor_state.state_representation import Point2D, StateSE2  # noqa: E402
from nuplan.common.actor_state.oriented_box import OrientedBox  # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402

ROOT = HERE / "results"
MODES = ["kinematic", "ocd"]


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def transform(origin, heading, points):
    points = np.asarray(points, dtype=float) - origin
    c, s = math.cos(heading), math.sin(heading)
    return np.column_stack((c * points[:, 0] + s * points[:, 1], -s * points[:, 0] + c * points[:, 1]))


def main():
    config = json.loads((HERE / "ocd_comparison_config.json").read_text())
    scenario, _ = build_scenario(config["modified_scenario_token"])
    data = {}
    for mode in MODES:
        directory = ROOT / mode
        rows = read_csv(directory / "ego_states.csv")
        data[mode] = {
            "rows": rows,
            "world": np.asarray([[float(r["rear_axle_x_m"]), float(r["rear_axle_y_m"])] for r in rows]),
            "headings": np.asarray([float(r["heading_rad"]) for r in rows]),
            "speeds": np.asarray([float(r["speed_mps"]) for r in rows]),
            "times": np.asarray([float(r["time_s"]) for r in rows]),
            "collision": np.maximum.accumulate([int(r["any_collision"]) for r in rows]).astype(bool),
            "references": np.load(directory / "reference_trajectories_global.npz")["trajectories"],
            "traffic": json.loads((directory / "traffic_tracks.json").read_text()),
            "stopped": json.loads((directory / "summary.json").read_text())["stopped_vehicle_poses"],
        }
    origin = data["kinematic"]["world"][0]
    initial_heading = data["kinematic"]["headings"][0]
    for mode in MODES:
        data[mode]["actual"] = transform(origin, initial_heading, data[mode]["world"])

    combined = np.vstack([data[m]["world"] for m in MODES])
    center = (combined.min(axis=0) + combined.max(axis=0)) / 2
    radius = float(np.linalg.norm(combined.max(axis=0) - combined.min(axis=0)) / 2 + 70)
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    nearby = scenario.map_api.get_proximal_map_objects(Point2D(*center), radius, layers)
    roads, centerlines, seen = [], [], set()
    for layer in layers:
        for obj in nearby[layer]:
            if obj.id in seen:
                continue
            seen.add(obj.id)
            roads.append(transform(origin, initial_heading, obj.polygon.exterior.coords))
            centerlines.append(transform(origin, initial_heading, [[p.x, p.y] for p in obj.baseline_path.discrete_path]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), dpi=100, sharey=True)
    fig.subplots_adjust(left=.055, right=.985, bottom=.16, top=.86, wspace=.08)
    artists = {}
    for ax, mode in zip(axes, MODES):
        ax.set_facecolor("#e6eee5")
        ax.add_collection(PolyCollection(roads, facecolors="#fbfbfb", edgecolors="#aab1b7", linewidths=.65, zorder=1))
        ax.add_collection(LineCollection(centerlines, colors="#bbc3c9", linewidths=.55, linestyles="dashed", zorder=2))
        agents = PolyCollection([], linewidths=.6, edgecolors="#6b4b25", zorder=5)
        ego = PolyCollection([], facecolors="#2563eb" if mode == "kinematic" else "#dc2626", edgecolors="#111827", zorder=8)
        ax.add_collection(agents)
        ax.add_collection(ego)
        driven, = ax.plot([], [], color="#2563eb" if mode == "kinematic" else "#dc2626", lw=2.6, zorder=6)
        planned, = ax.plot([], [], color="#00a6ad", lw=2.4, zorder=6)
        status = ax.text(.02, .97, "", transform=ax.transAxes, va="top", fontsize=10,
                         bbox=dict(facecolor="white", alpha=.93, edgecolor="none"), zorder=10)
        ax.set_title("nuPlan kinematic bicycle" if mode == "kinematic" else "Open Car Dynamics double track", fontsize=13)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Forward from initial pose (m)")
        artists[mode] = (agents, ego, driven, planned, status)
    axes[0].set_ylabel("Lateral from initial pose (m)")
    fig.suptitle("Same Flow Planner + LQR + interPlan scene | only vehicle model changes", fontsize=15)
    fig.legend(
        handles=[
            Line2D([], [], color="#00a6ad", lw=3, label="Current Flow Planner reference"),
            Line2D([], [], color="#2563eb", lw=3, label="Kinematic executed path"),
            Line2D([], [], color="#dc2626", lw=3, label="OCD executed path / stopped vehicle"),
            Line2D([], [], color="#ed9b40", lw=7, label="Reactive traffic"),
        ],
        loc="lower center", bbox_to_anchor=(.5, .025), ncol=4, frameon=False, fontsize=9,
    )
    vehicle = scenario.ego_vehicle_parameters

    def update(index):
        returned = []
        centers = []
        for ax, mode in zip(axes, MODES):
            d = data[mode]
            agents, ego, driven, planned, status = artists[mode]
            polygons, colors = [], []
            for obj in d["traffic"][index]["objects"]:
                box = OrientedBox(StateSE2(obj["x"], obj["y"], obj["heading"]), obj["length"], obj["width"], obj["height"])
                polygons.append(transform(origin, initial_heading, box.geometry.exterior.coords))
                stopped = any(math.hypot(obj["x"] - p[0], obj["y"] - p[1]) < .3 for p in d["stopped"])
                colors.append("#dc2626" if stopped else ("#ed9b40" if obj["type"] == "VEHICLE" else "#a855b5"))
            agents.set_verts(polygons)
            agents.set_facecolors(colors)
            footprint = CarFootprint.build_from_rear_axle(
                StateSE2(d["world"][index, 0], d["world"][index, 1], d["headings"][index]), vehicle
            )
            ego.set_verts([transform(origin, initial_heading, footprint.geometry.exterior.coords)])
            driven.set_data(d["actual"][:index + 1, 0], d["actual"][:index + 1, 1])
            future = transform(origin, initial_heading, d["references"][index, :, 1:3])
            planned.set_data(future[:, 0], future[:, 1])
            centers.append(d["actual"][index])
            status.set_text(
                f"t = {d['times'][index]:4.1f} s    speed = {d['speeds'][index]:4.2f} m/s\n"
                f"collision so far: {'YES' if d['collision'][index] else 'no'}"
            )
            status.set_color("#b91c1c" if d["collision"][index] else "#111827")
            returned.extend([agents, ego, driven, planned, status])
        center_x = float(np.mean([p[0] for p in centers]))
        center_y = float(np.mean([p[1] for p in centers]))
        for ax in axes:
            ax.set_xlim(center_x - 25, center_x + 65)
            ax.set_ylim(center_y - 22, center_y + 22)
        return returned

    indices = list(range(0, min(len(data[m]["rows"]) for m in MODES), 2))
    animation = FuncAnimation(fig, update, frames=indices, interval=200, blit=False)
    output = ROOT / "kinematic_vs_ocd_side_by_side.gif"
    animation.save(output, writer=PillowWriter(fps=5))
    update(indices[len(indices) // 2])
    fig.savefig(ROOT / "kinematic_vs_ocd_preview.png", dpi=140)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
