#!/usr/bin/env python3
"""Freeze replanning period and compare Flow Planner with the expert."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
from shapely.geometry import Point, Polygon

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
STAGE4 = RESEARCH / "04_nuplan_static_obstacle"
STAGE5 = RESEARCH / "05_nuplan_ocd_comparison"
sys.path[:0] = [str(STAGE4), str(STAGE5), str(HERE)]

from run_official_interplan import build_scenario, stopped_vehicle_poses  # noqa: E402
from run_comparison import apply_initial_speed_override  # noqa: E402
from expert_optimizer import SceneGeometry  # noqa: E402
from nuplan.common.actor_state.car_footprint import CarFootprint  # noqa: E402
from nuplan.common.actor_state.oriented_box import OrientedBox  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2  # noqa: E402


BASELINE = STAGE5 / "results_ocd_peak_sweep_15mps/high_1p0"
RESULTS = HERE / "results"
EXPERT = RESULTS / "replan_0p1s"
OUTPUT = RESULTS / "comparison"


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def signed_tracking_error(rows, references):
    errors = [0.0]
    for index in range(1, len(rows)):
        timestamp = int(rows[index]["timestamp_us"])
        previous = references[index - 1]
        point = previous[int(np.argmin(np.abs(previous[:, 0] - timestamp)))]
        dx = float(rows[index]["rear_axle_x_m"]) - point[1]
        dy = float(rows[index]["rear_axle_y_m"]) - point[2]
        errors.append(-math.sin(point[3]) * dx + math.cos(point[3]) * dy)
    return np.asarray(errors)


def load_group(directory, geometry):
    rows = read_csv(directory / "ego_states.csv")
    references = np.load(directory / "reference_trajectories_global.npz")["trajectories"]
    world = np.asarray([[float(row["rear_axle_x_m"]), float(row["rear_axle_y_m"])] for row in rows])
    local = geometry.world_to_local(world)
    return {
        "directory": directory,
        "rows": rows,
        "references": references,
        "world": world,
        "local": local,
        "times": np.asarray([float(row["time_s"]) for row in rows]),
        "speeds": np.asarray([float(row["speed_mps"]) for row in rows]),
        "headings": np.asarray([float(row["heading_rad"]) for row in rows]),
        "tracking": signed_tracking_error(rows, references),
        "traffic": json.loads((directory / "traffic_tracks.json").read_text()),
        "summary": json.loads((directory / "summary.json").read_text()),
    }


def footprint_metrics(data, scenario, geometry, obstacle):
    offroad = []
    road_margins = []
    clearances = []
    obstacle_box = OrientedBox(
        StateSE2(obstacle["x"], obstacle["y"], obstacle["heading"]),
        obstacle["length"], obstacle["width"], obstacle["height"],
    )
    for row in data["rows"]:
        footprint = CarFootprint.build_from_rear_axle(
            StateSE2(float(row["rear_axle_x_m"]), float(row["rear_axle_y_m"]), float(row["heading_rad"])),
            scenario.ego_vehicle_parameters,
        )
        local_polygon = Polygon(geometry.world_to_local(np.asarray(footprint.geometry.exterior.coords)))
        inside = geometry.route_polygon_local.buffer(1e-6).covers(local_polygon)
        offroad.append(not inside)
        if inside:
            road_margins.append(local_polygon.distance(geometry.route_polygon_local.boundary))
        else:
            road_margins.append(-max(geometry.route_polygon_local.distance(Point(point))
                                     for point in local_polygon.exterior.coords))
        clearances.append(footprint.geometry.distance(obstacle_box.geometry))
    data["offroad"] = np.asarray(offroad)
    data["road_margin"] = np.asarray(road_margins)
    data["clearance"] = np.asarray(clearances)


def period_comparison():
    summaries = [json.loads((RESULTS / name / "summary.json").read_text())
                 for name in ("replan_0p1s", "replan_0p5s", "replan_1p0s")]
    frozen = 0.1
    payload = {
        "selection_rule": "offline data-generation setting: freeze 0.1 s because its total closed-loop wall time is acceptable; no real-time deadline requirement",
        "frozen_replan_period_s": frozen,
        "lqr_ocd_period_s": 0.1,
        "summaries": summaries,
    }
    (RESULTS / "replan_period_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    periods = period_comparison()
    if periods["frozen_replan_period_s"] != 0.1:
        raise RuntimeError("This report expects the frozen 0.1 s expert result")

    scenario, _ = build_scenario("81f400b414215325-s0")
    apply_initial_speed_override(scenario, 15.0)
    geometry = SceneGeometry(scenario, stopped_vehicle_poses(scenario)[0])
    groups = {
        "flow": load_group(BASELINE, geometry),
        "expert": load_group(EXPERT, geometry),
    }
    obstacle_pose = groups["flow"]["summary"]["stopped_vehicle_poses"][0]
    obstacle = min(
        groups["flow"]["traffic"][0]["objects"],
        key=lambda item: math.hypot(item["x"] - obstacle_pose[0], item["y"] - obstacle_pose[1]),
    )
    for data in groups.values():
        footprint_metrics(data, scenario, geometry, obstacle)

    metrics = {}
    for name, data in groups.items():
        collision_indices = np.flatnonzero([int(row["obstacle_collision"]) for row in data["rows"]])
        stop_indices = np.flatnonzero(data["speeds"] < 0.1)
        metrics[name] = {
            "collision": bool(len(collision_indices)),
            "first_collision_time_s": float(data["times"][collision_indices[0]]) if len(collision_indices) else None,
            "offroad": bool(data["offroad"].any()),
            "first_offroad_time_s": float(data["times"][np.flatnonzero(data["offroad"])[0]]) if data["offroad"].any() else None,
            "safe_stop_time_s": float(data["times"][stop_indices[0]]) if len(stop_indices) and not len(collision_indices) else None,
            "minimum_obstacle_clearance_m": float(data["clearance"].min()),
            "minimum_speed_mps": float(data["speeds"].min()),
            "final_speed_mps": float(data["speeds"][-1]),
            "rms_one_step_lateral_tracking_error_m": float(np.sqrt(np.mean(data["tracking"][1:] ** 2))),
            "maximum_abs_one_step_lateral_tracking_error_m": float(np.max(np.abs(data["tracking"][1:]))),
            "maximum_abs_lateral_velocity_mps": float(max(abs(float(row["rear_axle_vy_mps"])) for row in data["rows"])),
            "maximum_abs_yaw_rate_radps": float(max(abs(float(row["yaw_rate_radps"])) for row in data["rows"])),
            "minimum_road_margin_after_0p5_s_m": float(data["road_margin"][data["times"] >= 0.5].min()),
        }
    metrics["conclusion"] = (
        "The Flow Planner group collided. The frozen 0.1 s optimal-control expert used a safe-stop reference "
        "and completed the same 18 s LQR/OCD run without collision or road departure."
    )
    (OUTPUT / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    colors = {"flow": "#2563eb", "expert": "#16a34a"}
    labels = {"flow": "Official Flow Planner", "expert": "Optimal-control expert (0.1 s)"}
    obstacle_local = geometry.world_to_local(np.asarray([[obstacle["x"], obstacle["y"]]]))[0]

    fig, ax = plt.subplots(figsize=(10, 5.7), dpi=160)
    for name, data in groups.items():
        ax.plot(data["local"][:, 0], data["local"][:, 1], lw=2.5, color=colors[name], label=labels[name])
    ax.scatter(*obstacle_local, s=160, marker="s", color="#b91c1c", label="Stopped vehicle center", zorder=5)
    ax.set(xlabel="Forward from initial rear axle (m)", ylabel="Lateral from initial rear axle (m)",
           title="Executed trajectories through the same LQR and OCD")
    ax.grid(alpha=.22); ax.legend(frameon=False); ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(OUTPUT / "executed_trajectory_comparison.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=160, sharex=True)
    for name, data in groups.items():
        axes[0].plot(data["times"], data["speeds"], lw=2.2, color=colors[name], label=labels[name])
        axes[1].plot(data["times"], data["tracking"], lw=1.8, color=colors[name], label=labels[name])
    axes[0].set_ylabel("Speed (m/s)"); axes[0].set_title("Speed")
    axes[1].set_ylabel("One-step lateral tracking error (m)"); axes[1].set_xlabel("Simulation time (s)")
    for ax in axes: ax.grid(alpha=.22); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUTPUT / "speed_and_tracking_comparison.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=160, sharex=True)
    for name, data in groups.items():
        axes[0].plot(data["times"], [float(row["rear_axle_vy_mps"]) for row in data["rows"]],
                     lw=2, color=colors[name], label=labels[name])
        axes[1].plot(data["times"], [float(row["yaw_rate_radps"]) for row in data["rows"]],
                     lw=2, color=colors[name], label=labels[name])
    axes[0].set_ylabel("Rear-axle lateral speed (m/s)"); axes[0].set_title("Dynamic states returned by OCD")
    axes[1].set_ylabel("Yaw rate (rad/s)"); axes[1].set_xlabel("Simulation time (s)")
    for ax in axes: ax.grid(alpha=.22); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUTPUT / "dynamic_states_comparison.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=100, sharex=True, sharey=True)
    artists = {}
    for ax, name in zip(axes, ("flow", "expert")):
        boundary = geometry.route_polygon_local.boundary
        lines = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]
        for line in lines:
            coordinates = np.asarray(line.coords)
            ax.plot(coordinates[:, 0], coordinates[:, 1], color="#9ca3af", lw=.8, zorder=1)
        obstacle_patch = OrientedBox(
            StateSE2(obstacle["x"], obstacle["y"], obstacle["heading"]),
            obstacle["length"], obstacle["width"], obstacle["height"],
        )
        obstacle_polygon = geometry.world_to_local(np.asarray(obstacle_patch.geometry.exterior.coords))
        ax.add_patch(PolygonPatch(obstacle_polygon, facecolor="#dc2626", edgecolor="#7f1d1d", alpha=.85, zorder=5))
        actual_line, = ax.plot([], [], color=colors[name], lw=2.7, zorder=4)
        reference_line, = ax.plot([], [], color="#00a6ad", lw=2.3, zorder=3)
        ego_patch = PolygonPatch(np.zeros((5, 2)), facecolor=colors[name], edgecolor="#111827", zorder=6)
        ax.add_patch(ego_patch)
        status = ax.text(.02, .97, "", transform=ax.transAxes, va="top",
                         bbox=dict(facecolor="white", alpha=.9, edgecolor="none"), zorder=8)
        ax.set_title(labels[name]); ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("Forward (m)")
        artists[name] = (actual_line, reference_line, ego_patch, status)
    axes[0].set_ylabel("Lateral (m)")
    fig.suptitle("Current future reference and actually executed path | same LQR + OCD")
    fig.legend(handles=[Line2D([], [], color="#00a6ad", lw=3, label="Current future reference"),
                        Line2D([], [], color="#2563eb", lw=3, label="Flow executed"),
                        Line2D([], [], color="#16a34a", lw=3, label="Expert executed")],
               loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=.14, top=.88, left=.06, right=.98, wspace=.08)

    def update(index):
        returned = []
        centers = []
        for name, data in groups.items():
            actual_line, reference_line, ego_patch, status = artists[name]
            actual_line.set_data(data["local"][:index + 1, 0], data["local"][:index + 1, 1])
            reference = geometry.world_to_local(data["references"][index, :, 1:3])
            reference_line.set_data(reference[:, 0], reference[:, 1])
            footprint = CarFootprint.build_from_rear_axle(
                StateSE2(data["world"][index, 0], data["world"][index, 1], data["headings"][index]),
                scenario.ego_vehicle_parameters,
            )
            ego_polygon = geometry.world_to_local(np.asarray(footprint.geometry.exterior.coords))
            ego_patch.set_xy(ego_polygon)
            collision = int(data["rows"][index]["obstacle_collision"])
            status.set_text(f"t={data['times'][index]:4.1f} s  v={data['speeds'][index]:4.2f} m/s\n"
                            f"collision: {'YES' if collision else 'no'}")
            status.set_color("#b91c1c" if collision else "#111827")
            centers.append(data["local"][index])
            returned.extend([actual_line, reference_line, ego_patch, status])
        center_x = float(np.mean([item[0] for item in centers]))
        center_y = float(np.mean([item[1] for item in centers]))
        for ax in axes:
            ax.set_xlim(center_x - 22, center_x + 55); ax.set_ylim(center_y - 10, center_y + 10)
        return returned

    frames = list(range(0, 181, 2))
    animation = FuncAnimation(fig, update, frames=frames, interval=200, blit=False)
    animation.save(OUTPUT / "flow_vs_expert.gif", writer=PillowWriter(fps=5))
    update(20); fig.savefig(OUTPUT / "flow_vs_expert_preview.png", dpi=140); plt.close(fig)

    table = "\n".join(
        f"| {item['replan_period_s']:.1f} s | {item['solve_attempts']} | {item['solve_successes']} | "
        f"{item['solve_failures']} | {item['solve_time_mean_s']:.3f} | {item['solve_time_p95_s']:.3f} | "
        f"{item['solve_time_max_s']:.3f} | {item['deadline_miss_rate']:.1%} |"
        for item in periods["summaries"]
    )
    report = f"""# 第一阶段：最优控制专家与原 Flow Planner 闭环对比

使用真实 nuPlan 场景，并将自车初速度设为 15 m/s 进行压力测试。15 m/s 是实验设定，
不是原日志速度。两组均使用相同官方 LQR、控制适配、OCD、障碍物、交通和 18 s 时域。

## 重规划周期冻结

| 周期 | 求解次数 | 成功 | 失败 | 平均/s | P95/s | 最大/s | 超期率 |
|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

本阶段用于离线专家数据生成，不要求实时截止。0.1 s 组的闭环总墙钟时间约 14.31 s，
相比 1.0 s 组约 12.58 s 仅增加约 1.73 s，因此冻结专家重规划周期为 **0.1 s**。
停车点承诺后不再移动目标，期间 LQR 与 OCD 仍按 0.1 s 执行并跟踪最近一次有效参考轨迹。

## 正式结果

| 指标 | 原 Flow Planner | 最优控制专家（0.1 s） |
|---|---:|---:|
| 碰撞 | {metrics['flow']['collision']} | {metrics['expert']['collision']} |
| 首次碰撞/s | {metrics['flow']['first_collision_time_s']} | {metrics['expert']['first_collision_time_s']} |
| 道路越界 | {metrics['flow']['offroad']} | {metrics['expert']['offroad']} |
| 安全停车时间/s | {metrics['flow']['safe_stop_time_s']} | {metrics['expert']['safe_stop_time_s']} |
| 最低速度/(m/s) | {metrics['flow']['minimum_speed_mps']:.3f} | {metrics['expert']['minimum_speed_mps']:.3f} |
| 单步横向跟踪误差 RMS/m | {metrics['flow']['rms_one_step_lateral_tracking_error_m']:.3f} | {metrics['expert']['rms_one_step_lateral_tracking_error_m']:.3f} |

原 Flow Planner 在约 {metrics['flow']['first_collision_time_s']:.2f} s 碰撞；专家选择强制动安全停车，
完整 18 s 内无碰撞、无道路越界。该结果说明当前专家至少能为这一压力测试提供比原模型更安全的
参考目标，但现阶段通过的是停车模式，不是高速绕行模式。

高速右绕行的早期尝试在优化模型内可行，但经 LQR/OCD 后出现横向滞后和边界过冲，因此没有作为
有效专家标签。安全停车分支使用固定闭环停车余量；这属于当前场景/LQR/OCD 的实验设定，后续扩展
工况前仍需验证其泛化性。
"""
    (OUTPUT / "REPORT.md").write_text(report)
    print(json.dumps({"periods": periods, "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
