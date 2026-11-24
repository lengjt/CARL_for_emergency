#!/usr/bin/env python3
"""Versioned expert-data serialization and closed-loop admission checks."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon

from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2


CONTRACT_VERSION = "expert_dataset_v1"
QUERY_STATE_COLUMNS = [
    "rear_axle_x_m", "rear_axle_y_m", "heading_rad", "rear_axle_vx_mps",
    "rear_axle_vy_mps", "yaw_rate_radps", "tire_steering_angle_rad",
]
DYNAMICS_CONDITION_COLUMNS = [
    "longitudinal_velocity_mps", "lateral_velocity_mps", "yaw_rate_radps",
    "tire_steering_angle_rad", "ocd_internal_mf52_peak_scale",
]
REFERENCE_LOCAL_COLUMNS = [
    "relative_time_s", "rear_axle_x_m", "rear_axle_y_m", "heading_rad", "speed_mps",
]
REFERENCE_GLOBAL_COLUMNS = [
    "timestamp_us", "rear_axle_x_m", "rear_axle_y_m", "heading_rad", "speed_mps",
]
EXECUTED_COLUMNS = [
    "relative_time_s", "timestamp_us", "rear_axle_x_m", "rear_axle_y_m", "heading_rad",
    "speed_mps", "rear_axle_vx_mps", "rear_axle_vy_mps", "yaw_rate_radps",
    "tire_steering_angle_rad", "road_margin_m", "obstacle_collision", "off_route_road",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def stable_id(prefix: str, payload: Any, length: int = 16) -> str:
    canonical = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{prefix}_{hashlib.sha256(canonical.encode()).hexdigest()[:length]}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def build_condition(scenario: Any, geometry: Any, config: dict[str, Any]) -> dict[str, Any]:
    ego = scenario.initial_ego_state
    vehicle = scenario.ego_vehicle_parameters
    obstacle_world = geometry.local_to_world(np.asarray([[geometry.obstacle_x, geometry.obstacle_y]]))[0]
    return {
        "contract_version": CONTRACT_VERSION,
        "scenario": {
            "modified_token": config["scenario_token"],
            "base_token": scenario.token,
            "scenario_type": scenario.scenario_type,
            "map_name": scenario.map_api.map_name,
        },
        "initial_ego_state": {
            "timestamp_us": ego.time_point.time_us,
            "rear_axle_x_m": ego.rear_axle.x,
            "rear_axle_y_m": ego.rear_axle.y,
            "heading_rad": ego.rear_axle.heading,
            "longitudinal_velocity_mps": ego.dynamic_car_state.rear_axle_velocity_2d.x,
            "lateral_velocity_mps": ego.dynamic_car_state.rear_axle_velocity_2d.y,
            "yaw_rate_radps": ego.dynamic_car_state.angular_velocity,
            "tire_steering_angle_rad": ego.tire_steering_angle,
        },
        "obstacles": [{
            "kind": "injected_stopped_vehicle",
            "center_x_m": obstacle_world[0],
            "center_y_m": obstacle_world[1],
            "heading_rad": geometry.heading + geometry.obstacle_heading,
            "length_m": geometry.obstacle_length,
            "width_m": geometry.obstacle_width,
        }],
        "vehicle": {
            "length_m": vehicle.length,
            "width_m": vehicle.width,
            "wheelbase_m": vehicle.wheel_base,
            "rear_axle_to_center_m": vehicle.rear_axle_to_center,
        },
        "environment": {
            "ocd_internal_mf52_peak_scale": config["ocd_internal_mf52_peak_scale"],
            "interpretation": "relative MF52 peak scaling parameter; not a calibrated friction coefficient",
        },
        "execution": {
            "expert_replan_period_s": config["expert_replan_period_s"],
            "lqr_ocd_period_s": config["lqr_ocd_period_s"],
            "tracker": "official nuPlan LQR, unchanged",
            "motion_model": "Open Car Dynamics double-track MF5.2",
            "reference_interface": "timestamped nuPlan InterpolatedTrajectory",
            "direct_expert_control_to_ocd": False,
        },
    }


def _reference_in_query_frame(reference: np.ndarray, query: np.ndarray, timestamp_us: int) -> np.ndarray:
    dx = reference[:, 1] - query[0]
    dy = reference[:, 2] - query[1]
    c, s = math.cos(query[2]), math.sin(query[2])
    result = np.empty_like(reference, dtype=np.float64)
    result[:, 0] = (reference[:, 0] - timestamp_us) * 1e-6
    result[:, 1] = c * dx + s * dy
    result[:, 2] = -s * dx + c * dy
    result[:, 3] = np.arctan2(np.sin(reference[:, 3] - query[2]), np.cos(reference[:, 3] - query[2]))
    result[:, 4] = reference[:, 4]
    return result


def _closed_loop_metrics(
    rows: list[dict[str, str]], scenario: Any, geometry: Any, maneuver_type: str,
) -> tuple[dict[str, Any], list[str]]:
    obstacle = geometry.local_to_world(np.asarray([[geometry.obstacle_x, geometry.obstacle_y]]))[0]
    obstacle_box = OrientedBox(
        StateSE2(obstacle[0], obstacle[1], geometry.heading + geometry.obstacle_heading),
        geometry.obstacle_length, geometry.obstacle_width, 1.5,
    )
    clearances, recalculated_offroad = [], []
    for row in rows:
        footprint = CarFootprint.build_from_rear_axle(
            StateSE2(float(row["rear_axle_x_m"]), float(row["rear_axle_y_m"]), float(row["heading_rad"])),
            scenario.ego_vehicle_parameters,
        )
        clearances.append(footprint.geometry.distance(obstacle_box.geometry))
        local_polygon = Polygon(geometry.world_to_local(np.asarray(footprint.geometry.exterior.coords)))
        recalculated_offroad.append(not geometry.route_polygon_local.buffer(1e-6).covers(local_polygon))

    world = np.asarray([[float(row["rear_axle_x_m"]), float(row["rear_axle_y_m"])] for row in rows])
    local = geometry.world_to_local(world)
    local_headings = np.asarray([
        math.atan2(
            math.sin(float(row["heading_rad"]) - geometry.heading),
            math.cos(float(row["heading_rad"]) - geometry.heading),
        ) for row in rows
    ])
    center_x = local[:, 0] + scenario.ego_vehicle_parameters.rear_axle_to_center * np.cos(local_headings)
    center_y = local[:, 1] + scenario.ego_vehicle_parameters.rear_axle_to_center * np.sin(local_headings)
    speeds = np.asarray([float(row["speed_mps"]) for row in rows])
    collisions = np.asarray([bool(int(row["obstacle_collision"])) for row in rows])
    offroad = np.asarray(recalculated_offroad) | np.asarray([bool(int(row["off_route_road"])) for row in rows])
    pass_threshold = geometry.obstacle_x + 0.5 * (5.176 + geometry.obstacle_length) + 5.0
    passed = bool(np.max(local[:, 0]) >= pass_threshold)
    stopped = bool(np.min(speeds) < 0.1 and speeds[-1] < 0.2)
    closest_index = int(np.argmin(np.abs(center_x - geometry.obstacle_x)))
    pass_lateral_offset = float(center_y[closest_index] - geometry.obstacle_y)
    detected_pass_side = "left" if pass_lateral_offset > 0 else "right"
    reasons = []
    if collisions.any():
        reasons.append("collision")
    if offroad.any():
        reasons.append("road_departure")
    if maneuver_type == "safe_stop" and not stopped:
        reasons.append("did_not_reach_safe_stop")
    if maneuver_type.startswith("pass_") and not passed:
        reasons.append("did_not_pass_obstacle")
    requested_pass_side = maneuver_type.removeprefix("pass_") if maneuver_type.startswith("pass_") else None
    if requested_pass_side is not None and passed and detected_pass_side != requested_pass_side:
        reasons.append("maneuver_side_mismatch")
    metrics = {
        "collision": bool(collisions.any()),
        "first_collision_time_s": (float(rows[int(np.flatnonzero(collisions)[0])]["time_s"])
                                   if collisions.any() else None),
        "road_departure": bool(offroad.any()),
        "first_road_departure_time_s": (float(rows[int(np.flatnonzero(offroad)[0])]["time_s"])
                                        if offroad.any() else None),
        "minimum_obstacle_clearance_m": float(np.min(clearances)),
        "minimum_speed_mps": float(np.min(speeds)),
        "final_speed_mps": float(speeds[-1]),
        "maximum_forward_progress_m": float(np.max(local[:, 0])),
        "passed_obstacle": passed,
        "requested_pass_side": requested_pass_side,
        "detected_pass_side": detected_pass_side if passed else None,
        "pass_lateral_offset_at_obstacle_m": pass_lateral_offset if passed else None,
        "reached_safe_stop": stopped,
        "pass_threshold_local_x_m": pass_threshold,
    }
    return metrics, reasons


def package_solution(
    raw_directory: Path,
    solution_directory: Path,
    condition_id: str,
    condition: dict[str, Any],
    maneuver_type: str,
    scenario: Any,
    geometry: Any,
) -> dict[str, Any]:
    summary = json.loads((raw_directory / "summary.json").read_text())
    rows = read_csv(raw_directory / "ego_states.csv")
    calls = read_csv(raw_directory / "planner_call_log.csv")
    solves = read_csv(raw_directory / "expert_solve_log.csv")
    references = np.load(raw_directory / "reference_trajectories_global.npz")["trajectories"]
    if not (len(rows) == len(calls) == len(references)):
        raise ValueError(f"Unaligned rollout records in {raw_directory}")

    solution_id = stable_id("solution", {"condition_id": condition_id, "maneuver_type": maneuver_type})
    successful_solves = {int(row["timestamp_us"]): row for row in solves if row["success"] == "True"}
    selected = [index for index, call in enumerate(calls) if int(call["timestamp_us"]) in successful_solves]
    query_states, dynamic_conditions, local_references, global_references, sample_ids = [], [], [], [], []
    solver_records = []
    peak_scale = float(condition["environment"]["ocd_internal_mf52_peak_scale"])
    for index in selected:
        row = rows[index]
        timestamp_us = int(row["timestamp_us"])
        query = np.asarray([float(row[column]) for column in QUERY_STATE_COLUMNS], dtype=np.float64)
        query_states.append(query)
        dynamic_conditions.append([query[3], query[4], query[5], query[6], peak_scale])
        global_reference = np.asarray(references[index], dtype=np.float64)
        global_references.append(global_reference)
        local_references.append(_reference_in_query_frame(global_reference, query, timestamp_us))
        sample_id = stable_id("sample", {
            "condition_id": condition_id, "solution_id": solution_id, "timestamp_us": timestamp_us,
        })
        sample_ids.append(sample_id)
        solve = successful_solves[timestamp_us]
        solver_records.append({
            "sample_id": sample_id,
            "timestamp_us": timestamp_us,
            "solve_time_s": float(solve["solve_time_s"]),
            "iterations": int(solve["iterations"]),
            "status": solve["status"],
            "objective": float(solve["objective"]) if solve["objective"] else None,
            "fresh_successful_solution": True,
        })

    solution_directory.mkdir(parents=True, exist_ok=True)
    sample_count = len(selected)
    reference_shape = references.shape[1:]
    np.savez_compressed(
        solution_directory / "reference_labels.npz",
        sample_ids=np.asarray(sample_ids, dtype="U40"),
        query_timestamp_us=np.asarray([int(rows[index]["timestamp_us"]) for index in selected], dtype=np.int64),
        query_state_global=np.asarray(query_states, dtype=np.float64).reshape(sample_count, len(QUERY_STATE_COLUMNS)),
        dynamics_condition=np.asarray(dynamic_conditions, dtype=np.float64).reshape(
            sample_count, len(DYNAMICS_CONDITION_COLUMNS)
        ),
        reference_local=np.asarray(local_references, dtype=np.float64).reshape(sample_count, *reference_shape),
        reference_global=np.asarray(global_references, dtype=np.float64).reshape(sample_count, *reference_shape),
        query_state_columns=np.asarray(QUERY_STATE_COLUMNS),
        dynamics_condition_columns=np.asarray(DYNAMICS_CONDITION_COLUMNS),
        reference_local_columns=np.asarray(REFERENCE_LOCAL_COLUMNS),
        reference_global_columns=np.asarray(REFERENCE_GLOBAL_COLUMNS),
    )
    with (solution_directory / "sample_solver_metadata.jsonl").open("w") as stream:
        for record in solver_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    executed = np.asarray([
        [float(row["time_s"]), int(row["timestamp_us"])]
        + [float(row[column]) for column in EXECUTED_COLUMNS[2:11]]
        + [int(row["obstacle_collision"]), int(row["off_route_road"])]
        for row in rows
    ], dtype=np.float64)
    np.savez_compressed(
        solution_directory / "executed_trajectory.npz",
        trajectory=executed,
        columns=np.asarray(EXECUTED_COLUMNS),
    )

    metrics, rejection_reasons = _closed_loop_metrics(rows, scenario, geometry, maneuver_type)
    if not selected:
        rejection_reasons.append("no_successful_expert_solution")
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "condition_id": condition_id,
        "solution_id": solution_id,
        "maneuver_type": maneuver_type,
        "admitted_as_training_label": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "fresh_reference_label_count": sample_count,
        "closed_loop_metrics": metrics,
        "solver": {
            "attempts": summary["solve_attempts"],
            "successes": summary["solve_successes"],
            "failures": summary["solve_failures"],
            "mean_solve_time_s": summary["solve_time_mean_s"],
            "p95_solve_time_s": summary["solve_time_p95_s"],
            "maximum_solve_time_s": summary["solve_time_max_s"],
        },
        "interfaces": {
            "reference": summary["reference_interface"],
            "execution_chain": summary["execution_chain"],
            "direct_control_to_ocd": summary["direct_control_to_ocd"],
        },
        "raw_run_directory": str(raw_directory.resolve()),
    }
    write_json(solution_directory / "solution.json", manifest)
    return manifest
