#!/usr/bin/env python3
"""Analyze and render the discriminating 12 m/s pass case."""
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

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path[:0] = [str(RESEARCH / "04_nuplan_static_obstacle"),
                str(RESEARCH / "05_nuplan_ocd_comparison"), str(HERE)]

from analyze_and_render import load_group, footprint_metrics  # noqa: E402
from run_official_interplan import build_scenario, stopped_vehicle_poses  # noqa: E402
from run_comparison import apply_initial_speed_override  # noqa: E402
from expert_optimizer import SceneGeometry  # noqa: E402
from nuplan.common.actor_state.car_footprint import CarFootprint  # noqa: E402
from nuplan.common.actor_state.oriented_box import OrientedBox  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2  # noqa: E402


ROOT = HERE / "results_scan"
FLOW = ROOT / "flow_v12_s1p0"
EXPERT = ROOT / "expert_v12_s1p0"
OUTPUT = ROOT / "comparison_v12_s1p0"


def solve_log(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario, _ = build_scenario("81f400b414215325-s0")
    apply_initial_speed_override(scenario, 12.0)
    geometry = SceneGeometry(scenario, stopped_vehicle_poses(scenario)[0])
    groups = {"flow": load_group(FLOW, geometry), "expert": load_group(EXPERT, geometry)}
    obstacle_pose = groups["flow"]["summary"]["stopped_vehicle_poses"][0]
    obstacle = min(groups["flow"]["traffic"][0]["objects"],
                   key=lambda item: math.hypot(item["x"] - obstacle_pose[0], item["y"] - obstacle_pose[1]))
    for data in groups.values():
        footprint_metrics(data, scenario, geometry, obstacle)

    expert_solves = solve_log(EXPERT / "expert_solve_log.csv")
    metrics = {}
    for name, data in groups.items():
        collision_indices = np.flatnonzero([int(row["obstacle_collision"]) for row in data["rows"]])
        threshold = geometry.obstacle_x + 0.5 * (5.176 + geometry.obstacle_length) + 5.0
        metrics[name] = {
            "collision": bool(len(collision_indices)),
            "first_collision_time_s": float(data["times"][collision_indices[0]]) if len(collision_indices) else None,
            "offroad": bool(data["offroad"].any()),
            "passed_obstacle": bool(data["local"][:, 0].max() >= threshold),
            "maximum_forward_progress_m": float(data["local"][:, 0].max()),
            "minimum_obstacle_clearance_m": float(data["clearance"].min()),
            "minimum_speed_mps": float(data["speeds"].min()),
            "final_speed_mps": float(data["speeds"][-1]),
            "rms_one_step_lateral_tracking_error_m": float(np.sqrt(np.mean(data["tracking"][1:] ** 2))),
            "maximum_abs_one_step_lateral_tracking_error_m": float(np.max(np.abs(data["tracking"][1:]))),
            "maximum_abs_lateral_velocity_mps": float(max(abs(float(row["rear_axle_vy_mps"])) for row in data["rows"])),
            "maximum_abs_yaw_rate_radps": float(max(abs(float(row["yaw_rate_radps"])) for row in data["rows"])),
        }
    metrics["expert_solver"] = {
        "attempts": len(expert_solves),
        "successes": sum(row["success"] == "True" for row in expert_solves),
        "failures": sum(row["success"] != "True" for row in expert_solves),
        "total_solve_time_s": float(sum(float(row["solve_time_s"]) for row in expert_solves)),
        "mean_solve_time_s": groups["expert"]["summary"]["solve_time_mean_s"],
        "p95_solve_time_s": groups["expert"]["summary"]["solve_time_p95_s"],
        "maximum_solve_time_s": groups["expert"]["summary"]["solve_time_max_s"],
        "closed_loop_wall_time_s": groups["expert"]["summary"]["wall_time_s"],
        "failure_policy": "keep tracking the most recent valid time-parameterized reference",
    }
    (OUTPUT / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    checks = {
        "same_initial_timestamp": groups["flow"]["rows"][0]["timestamp_us"] == groups["expert"]["rows"][0]["timestamp_us"],
        "same_initial_pose": float(np.max(np.abs(groups["flow"]["world"][0] - groups["expert"]["world"][0]))) < 1e-10,
        "same_initial_speed_12_mps": abs(groups["flow"]["speeds"][0] - 12.0) < 1e-12
        and abs(groups["expert"]["speeds"][0] - 12.0) < 1e-12,
        "same_181_frames": len(groups["flow"]["rows"]) == len(groups["expert"]["rows"]) == 181,
        "same_reference_shape": groups["flow"]["references"].shape == groups["expert"]["references"].shape == (181, 81, 5),
        "same_peak_scale": groups["flow"]["summary"]["ocd_vehicle"]["internal_peak_scale"]
        == groups["expert"]["summary"]["internal_peak_scale"] == 1.0,
        "same_obstacle_pose": np.max(np.abs(
            np.asarray(groups["flow"]["summary"]["stopped_vehicle_poses"][0])
            - np.asarray([geometry.local_to_world(np.asarray([[geometry.obstacle_x, geometry.obstacle_y]]))[0, 0],
                          geometry.local_to_world(np.asarray([[geometry.obstacle_x, geometry.obstacle_y]]))[0, 1],
                          stopped_vehicle_poses(scenario)[0].heading]))) < 1e-8,
        "expert_uses_lqr_ocd": groups["expert"]["summary"]["direct_control_to_ocd"] is False,
        "expert_passes_safely": metrics["expert"]["passed_obstacle"] and not metrics["expert"]["collision"]
        and not metrics["expert"]["offroad"],
        "flow_fails_same_case": metrics["flow"]["collision"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    verification = {"passed": all(checks.values()), "checks": checks}
    (OUTPUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    if not verification["passed"]:
        raise RuntimeError(verification)

    colors = {"flow": "#2563eb", "expert": "#16a34a"}
    labels = {"flow": "Official Flow Planner", "expert": "Optimal-control expert, 0.1 s"}
    obstacle_local = geometry.world_to_local(np.asarray([[obstacle["x"], obstacle["y"]]]))[0]

    fig, ax = plt.subplots(figsize=(10, 5.7), dpi=160)
    for name, data in groups.items():
        ax.plot(data["local"][:, 0], data["local"][:, 1], lw=2.5, color=colors[name], label=labels[name])
    ax.scatter(*obstacle_local, marker="s", s=160, color="#dc2626", label="Stopped vehicle center", zorder=5)
    ax.set(xlabel="Forward from initial rear axle (m)", ylabel="Lateral from initial rear axle (m)",
           title="12 m/s, peak scale 1.0 | executed through the same LQR and OCD")
    ax.grid(alpha=.22); ax.legend(frameon=False); ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(OUTPUT / "executed_trajectory_comparison.png"); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), dpi=160, sharex=True)
    for name, data in groups.items():
        axes[0].plot(data["times"], data["speeds"], lw=2.1, color=colors[name], label=labels[name])
        axes[1].plot(data["times"], data["tracking"], lw=1.8, color=colors[name], label=labels[name])
        axes[2].plot(data["times"], data["clearance"], lw=2.0, color=colors[name], label=labels[name])
    axes[0].set_ylabel("Speed (m/s)"); axes[0].set_title("Closed-loop comparison")
    axes[1].set_ylabel("One-step lateral error (m)")
    axes[2].set_ylabel("Obstacle clearance (m)"); axes[2].set_xlabel("Simulation time (s)")
    for ax in axes: ax.grid(alpha=.22); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUTPUT / "speed_tracking_clearance.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=100, sharex=True, sharey=True)
    artists = {}
    obstacle_box = OrientedBox(StateSE2(obstacle["x"], obstacle["y"], obstacle["heading"]),
                                obstacle["length"], obstacle["width"], obstacle["height"])
    obstacle_polygon = geometry.world_to_local(np.asarray(obstacle_box.geometry.exterior.coords))
    for ax, name in zip(axes, ("flow", "expert")):
        boundary = geometry.route_polygon_local.boundary
        lines = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]
        for line in lines:
            coordinates = np.asarray(line.coords)
            ax.plot(coordinates[:, 0], coordinates[:, 1], color="#9ca3af", lw=.8, zorder=1)
        ax.add_patch(PolygonPatch(obstacle_polygon, facecolor="#dc2626", edgecolor="#7f1d1d", zorder=5))
        driven, = ax.plot([], [], color=colors[name], lw=2.7, zorder=4)
        planned, = ax.plot([], [], color="#00a6ad", lw=2.3, zorder=3)
        ego = PolygonPatch(np.zeros((5, 2)), facecolor=colors[name], edgecolor="#111827", zorder=6)
        ax.add_patch(ego)
        status = ax.text(.02, .97, "", transform=ax.transAxes, va="top",
                         bbox=dict(facecolor="white", alpha=.9, edgecolor="none"), zorder=8)
        ax.set_title(labels[name]); ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("Forward (m)")
        artists[name] = driven, planned, ego, status
    axes[0].set_ylabel("Lateral (m)")
    fig.suptitle("12 m/s discriminating case | current future reference and executed path")
    fig.legend(handles=[Line2D([], [], color="#00a6ad", lw=3, label="Current future reference"),
                        Line2D([], [], color="#2563eb", lw=3, label="Flow executed"),
                        Line2D([], [], color="#16a34a", lw=3, label="Expert executed")],
               loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=.14, top=.88, left=.06, right=.98, wspace=.08)

    def update(index):
        returned, centers = [], []
        for name, data in groups.items():
            driven, planned, ego, status = artists[name]
            driven.set_data(data["local"][:index + 1, 0], data["local"][:index + 1, 1])
            reference = geometry.world_to_local(data["references"][index, :, 1:3])
            planned.set_data(reference[:, 0], reference[:, 1])
            footprint = CarFootprint.build_from_rear_axle(
                StateSE2(data["world"][index, 0], data["world"][index, 1], data["headings"][index]),
                scenario.ego_vehicle_parameters,
            )
            ego.set_xy(geometry.world_to_local(np.asarray(footprint.geometry.exterior.coords)))
            collision = int(data["rows"][index]["obstacle_collision"])
            status.set_text(f"t={data['times'][index]:4.1f} s  v={data['speeds'][index]:4.2f} m/s\n"
                            f"collision: {'YES' if collision else 'no'}")
            status.set_color("#b91c1c" if collision else "#111827")
            centers.append(data["local"][index]); returned.extend([driven, planned, ego, status])
        center_x = float(np.mean([point[0] for point in centers])); center_y = float(np.mean([point[1] for point in centers]))
        for ax in axes:
            ax.set_xlim(center_x - 22, center_x + 55); ax.set_ylim(center_y - 10, center_y + 10)
        return returned

    animation = FuncAnimation(fig, update, frames=list(range(0, 181, 2)), interval=200, blit=False)
    animation.save(OUTPUT / "flow_vs_expert_pass.gif", writer=PillowWriter(fps=5))
    update(34); fig.savefig(OUTPUT / "flow_vs_expert_pass_preview.png", dpi=140); plt.close(fig)

    report = f"""# 12 m/s 绕行区分工况

使用真实 nuPlan 场景，自车初速度设为 12 m/s，OCD 内部 MF5.2 峰值缩放为 1.0。
两组使用相同初态、障碍、交通、官方 LQR、控制适配、OCD 和 18 s 时域。

| 指标 | 原 Flow Planner | 最优控制专家 |
|---|---:|---:|
| 碰撞 | {metrics['flow']['collision']} | {metrics['expert']['collision']} |
| 首次碰撞/s | {metrics['flow']['first_collision_time_s']} | {metrics['expert']['first_collision_time_s']} |
| 道路越界 | {metrics['flow']['offroad']} | {metrics['expert']['offroad']} |
| 完成绕行 | {metrics['flow']['passed_obstacle'] and not metrics['flow']['collision']} | {metrics['expert']['passed_obstacle']} |
| 最小障碍间隙/m | {metrics['flow']['minimum_obstacle_clearance_m']:.3f} | {metrics['expert']['minimum_obstacle_clearance_m']:.3f} |
| 最大前向进度/m | {metrics['flow']['maximum_forward_progress_m']:.3f} | {metrics['expert']['maximum_forward_progress_m']:.3f} |

原 Flow Planner 在约 {metrics['flow']['first_collision_time_s']:.2f} s 碰撞。最优控制专家输出带时间信息的
nuPlan 参考轨迹，经相同 LQR + OCD 后完成绕行，无碰撞、无道路越界。因此该工况满足第一阶段要找的
区分条件：专家闭环可行，而原模型闭环失败。

专家按 0.1 s 周期共尝试 {metrics['expert_solver']['attempts']} 次求解，成功
{metrics['expert_solver']['successes']} 次、失败 {metrics['expert_solver']['failures']} 次，累计求解约
{metrics['expert_solver']['total_solve_time_s']:.1f} s，完整闭环墙钟时间约
{metrics['expert_solver']['closed_loop_wall_time_s']:.1f} s。失败周期继续跟踪最近一次有效参考；保存的
`planner_call_log.csv` 与 `expert_solve_log.csv` 可区分新解和复用参考。

原 Flow Planner 本次因 GPU 驱动不可用，在 CPU 上使用相同官方权重、seed 0、CFG 1.0 和四步采样完成；
规划输出和后续 LQR/OCD 链路未改变。
"""
    (OUTPUT / "REPORT.md").write_text(report)
    print(json.dumps({"verification": verification, "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
