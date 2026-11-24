"""Render recorded mini closed-loop simulations; no inference or training."""
import argparse
import json
import os
from pathlib import Path

from common import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.lines import Line2D
import numpy as np
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.simulation_log import SimulationLog


def render(log_path, output, score):
    # Only load trusted logs produced by our own run: nuPlan logs contain pickle.
    log = SimulationLog.load_data(log_path)
    samples = log.simulation_history.data
    start = samples[0].ego_state.rear_axle
    origin = np.array([start.x, start.y])
    c, s = np.cos(start.heading), np.sin(start.heading)
    rotation = np.array([[c, -s], [s, c]])

    def xy(points):
        return (np.asarray(points) - origin) @ rotation

    def path(states):
        return xy([[state.rear_axle.x, state.rear_axle.y] for state in states])

    actual = path([sample.ego_state for sample in samples])
    expert = path(list(log.scenario.get_expert_ego_trajectory()))
    world = np.array([[sample.ego_state.rear_axle.x, sample.ego_state.rear_axle.y]
                      for sample in samples])
    center = (world.min(axis=0) + world.max(axis=0)) / 2
    radius = float(np.linalg.norm(world.max(axis=0) - world.min(axis=0)) / 2 + 100)
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    objects = log.simulation_history.map_api.get_proximal_map_objects(
        Point2D(*center), radius, layers)
    roads, centerlines = [], []
    for layer in layers:
        for obj in objects[layer]:
            roads.append(xy(obj.polygon.exterior.coords))
            centerlines.append(xy([[p.x, p.y] for p in obj.baseline_path.discrete_path]))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.subplots_adjust(left=.09, right=.98, bottom=.19, top=.84)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#e4ece2")
    ax.add_collection(PolyCollection(roads, facecolors="#fafafa", edgecolors="#b7bdc3",
                                     linewidths=.6, zorder=1))
    ax.add_collection(LineCollection(centerlines, colors="#c5cbd1", linewidths=.5,
                                     linestyles="dashed", zorder=2))
    ax.plot(expert[:, 0], expert[:, 1], color="#697586", ls="--", lw=1.5, zorder=3)
    others = PolyCollection([], linewidths=.5, edgecolors="#825526", zorder=4)
    ego = PolyCollection([], facecolors="#2563eb", edgecolors="#153875", zorder=7)
    ax.add_collection(others)
    ax.add_collection(ego)
    trail, = ax.plot([], [], color="#2563eb", lw=2, zorder=5)
    planned, = ax.plot([], [], color="#00a6ad", lw=2.5, zorder=6)
    status = ax.text(.02, .97, "", transform=ax.transAxes, va="top", fontsize=11,
                     bbox=dict(facecolor="white", alpha=.92, edgecolor="none"), zorder=9)
    scenario_type = log.scenario.scenario_type
    fig.suptitle(f"Flow Planner | {scenario_type}\n"
                 f"Recorded closed-loop simulation | score {score:.3f}", fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forward distance from initial pose (m)")
    ax.set_ylabel("Lateral distance (m)")
    legend = [Line2D([], [], color=color, lw=3, label=label, ls=style)
              for color, label, style in [
                  ("#2563eb", "Ego / driven path", "-"),
                  ("#00a6ad", "Planned future", "-"),
                  ("#ed9b40", "Other vehicles", "-"),
                  ("#a855b5", "Pedestrians / cyclists", "-"),
                  ("#697586", "Human reference", "--")]]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .04),
               ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .015, "Official weights, no retraining | Surrounding traffic follows recorded logs",
             ha="center", fontsize=9, color="#526075")
    t0 = samples[0].iteration.time_s

    def update(index):
        sample = samples[index]
        polygons, colors = [], []
        for obj in sample.observation.tracked_objects.tracked_objects:
            polygons.append(xy(obj.box.geometry.exterior.coords))
            kind = obj.tracked_object_type.name
            colors.append("#ed9b40" if kind == "VEHICLE" else
                          "#a855b5" if kind in ("PEDESTRIAN", "BICYCLE") else "#89949e")
        others.set_verts(polygons)
        others.set_facecolors(colors)
        ego.set_verts([xy(sample.ego_state.car_footprint.geometry.exterior.coords)])
        trail.set_data(actual[:index + 1, 0], actual[:index + 1, 1])
        future = path(sample.trajectory.get_sampled_trajectory())
        planned.set_data(future[:, 0], future[:, 1])
        x, y = actual[index]
        ax.set_xlim(x - 30, x + 80)
        ax.set_ylim(y - 25, y + 25)
        status.set_text(f"t = {sample.iteration.time_s - t0:4.1f} s    "
                        f"speed = {sample.ego_state.dynamic_car_state.speed * 3.6:4.1f} km/h")
        return others, ego, trail, planned, status

    # 10 Hz logs sampled at 5 Hz; playback uses simulation time, not inference time.
    indices = list(range(0, len(samples), 2))
    # Round timestamp jitter so GIF centisecond quantization does not turn a
    # nominal 200 ms frame into 190 ms and noticeably speed up the replay.
    fps = round(1 / (2 * log.simulation_history.interval_seconds), 1)
    animation = FuncAnimation(fig, update, frames=indices, interval=1000 / fps, blit=False)
    destination = output / f"{scenario_type}.gif"
    animation.save(destination, writer=PillowWriter(fps=fps))
    update(indices[len(indices) // 2])
    fig.savefig(output / f"{scenario_type}.png", dpi=100)
    plt.close(fig)
    return {"scenario_type": scenario_type, "source": str(log_path),
            "gif": str(destination), "simulation_frames": len(samples),
            "rendered_frames": len(indices), "fps": fps,
            "duration_seconds": samples[-1].iteration.time_s - t0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts/nuplan/mini_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/nuplan/replays")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = []
    for scenario in summary["scenarios"]:
        paths = list((Path(summary["run_directory"]) / "simulation_log").glob(
            f"*/{scenario['scenario_type']}/*/{scenario['scenario']}/*.msgpack.xz"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one log for {scenario['scenario']}: {paths}")
        result = render(paths[0], args.output, scenario["score"])
        metadata.append(result)
        print(json.dumps(result), flush=True)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
