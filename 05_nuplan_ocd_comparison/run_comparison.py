#!/usr/bin/env python3
"""Run the same official interPlan nudge scene with kinematic and OCD motion models."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
STAGE1 = RESEARCH / "04_nuplan_static_obstacle"
sys.path.insert(0, str(STAGE1))

from run_official_interplan import (  # noqa: E402
    CHECKPOINT,
    INFERENCE_CONFIG,
    IDMAgents,
    LoggingController,
    Simulation,
    SimulationSetup,
    StepSimulationTimeController,
    build_controller,
    build_planner,
    build_scenario,
    save_csv,
    sha256,
    stopped_vehicle_poses,
    trajectory_array,
)
from ocd_motion_model import OCDMotionModel, OCD_PROJECT, PREFIX  # noqa: E402
from nuplan.common.actor_state.ego_state import EgoState  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D  # noqa: E402

CONFIG_PATH = HERE / "ocd_comparison_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
RESULTS = HERE / "results"


class FixedHorizonSimulation(Simulation):
    """Preserve the first-stage 18 s horizon instead of ending at a path-dependent goal time."""

    def __init__(self, *args, fixed_horizon_steps: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_horizon_steps = int(fixed_horizon_steps)

    def goal_reached(self, ego_state):
        return len(self.history.data) >= self.fixed_horizon_steps


def apply_initial_speed_override(scenario, target_speed_mps: float):
    """Create a speed-consistent 2 s ego history on the logged path, ending at the real initial pose."""
    target = float(target_speed_mps)
    if target <= 0:
        raise ValueError("Initial speed override must be positive")
    original_initial = scenario.initial_ego_state
    original_get_past = scenario.get_ego_past_trajectory
    original_get_state = scenario.get_ego_state_at_iteration
    initial = EgoState.build_from_rear_axle(
        rear_axle_pose=original_initial.rear_axle,
        rear_axle_velocity_2d=StateVector2D(target, 0.0),
        rear_axle_acceleration_2d=original_initial.dynamic_car_state.rear_axle_acceleration_2d,
        tire_steering_angle=original_initial.tire_steering_angle,
        time_point=original_initial.time_point,
        vehicle_parameters=original_initial.car_footprint.vehicle_parameters,
        is_in_auto_mode=True,
        angular_vel=original_initial.dynamic_car_state.angular_velocity,
        angular_accel=original_initial.dynamic_car_state.angular_acceleration,
        tire_steering_rate=original_initial.dynamic_car_state.tire_steering_rate,
    )
    scenario.__dict__["initial_ego_state"] = initial

    def get_state_at_iteration(this, iteration):
        return this.initial_ego_state if iteration == 0 else original_get_state(iteration)

    def get_speed_adjusted_past(this, iteration, time_horizon, num_samples=None):
        original = list(original_get_past(iteration, time_horizon, num_samples))
        if iteration != 0 or not original:
            return iter(original)
        adjusted = [None] * len(original)
        next_state = this.initial_ego_state
        for index in range(len(original) - 1, -1, -1):
            old = original[index]
            dt = (next_state.time_point.time_us - old.time_point.time_us) * 1e-6
            heading_delta = math.atan2(
                math.sin(next_state.rear_axle.heading - old.rear_axle.heading),
                math.cos(next_state.rear_axle.heading - old.rear_axle.heading),
            )
            midpoint_heading = old.rear_axle.heading + 0.5 * heading_delta
            pose = StateSE2(
                next_state.rear_axle.x - target * dt * math.cos(midpoint_heading),
                next_state.rear_axle.y - target * dt * math.sin(midpoint_heading),
                old.rear_axle.heading,
            )
            yaw_rate = heading_delta / dt if dt > 0 else 0.0
            adjusted[index] = EgoState.build_from_rear_axle(
                rear_axle_pose=pose,
                rear_axle_velocity_2d=StateVector2D(target, 0.0),
                rear_axle_acceleration_2d=StateVector2D(0.0, target * yaw_rate),
                tire_steering_angle=old.tire_steering_angle,
                time_point=old.time_point,
                vehicle_parameters=old.car_footprint.vehicle_parameters,
                is_in_auto_mode=True,
                angular_vel=yaw_rate,
                angular_accel=0.0,
                tire_steering_rate=old.dynamic_car_state.tire_steering_rate,
            )
            next_state = adjusted[index]
        return iter(adjusted)

    scenario.get_ego_state_at_iteration = types.MethodType(get_state_at_iteration, scenario)
    scenario.get_ego_past_trajectory = types.MethodType(get_speed_adjusted_past, scenario)
    return {
        "target_speed_mps": target,
        "original_speed_mps": original_initial.dynamic_car_state.speed,
        "history_method": "2 s ego history reconstructed backward on logged headings at constant target speed",
    }


def build_observation(scenario):
    return IDMAgents(
        target_velocity=10,
        min_gap_to_lead_agent=1.0,
        headway_time=1.5,
        accel_max=1.0,
        decel_max=2.0,
        open_loop_detections_types=["BARRIER", "CZONE_SIGN", "TRAFFIC_CONE", "GENERIC_OBJECT"],
        scenario=scenario,
        minimum_path_length=20,
        planned_trajectory_samples=None,
        planned_trajectory_sample_interval=None,
        radius=100,
        IDM_agents_behavior=scenario.modification.get("observation", "ASSERTIVE").upper(),
    )


def ocd_vehicle_metadata(internal_peak_scale=None):
    config = json.loads((OCD_PROJECT / "configs/vehicle.json").read_text())
    p = config["ocd_parameters"]
    wheel_mass = 2 * p[PREFIX + "mass_wheel_kg.front"] + 2 * p[PREFIX + "mass_wheel_kg.rear"]
    return {
        "name": config["name"],
        "status": config["status"],
        "body_length_m": config["body"]["length_m"],
        "body_width_m": config["body"]["width_m"],
        "vehicle_mass_kg": p[PREFIX + "mass_vehicle_kg"],
        "wheel_mass_total_kg": wheel_mass,
        "force_adapter_effective_mass_kg": p[PREFIX + "mass_vehicle_kg"] + wheel_mass,
        "wheelbase_m": p[PREFIX + "wheelbase_m"],
        "cg_from_front_axle_m": p[PREFIX + "cog.distance_from_front_axle_m"],
        "cg_from_rear_axle_m": p[PREFIX + "wheelbase_m"] - p[PREFIX + "cog.distance_from_front_axle_m"],
        "steering_pt1_s": p["steering_actuator.T_PT1"],
        "steering_angle_limit_rad": p["steering_actuator.angle_max_rad"],
        "steering_rate_limit_radps": p["steering_actuator.angle_rate_max_radps"],
        "mf52_lmux": p[PREFIX + "z.tire_model.front_left.LMUX"],
        "mf52_lmuy": p[PREFIX + "z.tire_model.front_left.LMUY"],
        "internal_peak_scale": (
            CONFIG["ocd"]["internal_mf52_peak_scale"]
            if internal_peak_scale is None
            else float(internal_peak_scale)
        ),
        "internal_peak_scale_interpretation": "relative MF52 LMUX/LMUY scale, not calibrated road friction",
        "command_delay_s": config["execution"]["command_delay_s"],
        "force_time_constant_s": config["execution"]["force_time_constant_s"],
        "force_rate_limit_Nps": config["execution"]["force_rate_Nps"],
        "max_drive_force_N": config["execution"]["max_drive_force_N"],
        "max_brake_force_N": config["execution"]["max_brake_force_N"],
    }


def run(
    mode: str,
    force: bool = False,
    output_root: Path = RESULTS,
    output_name: str | None = None,
    ocd_peak_scale: float | None = None,
    initial_speed_mps: float | None = None,
    planner_device: str = "cuda",
    planner_config_path: Path | None = None,
):
    result_id = output_name or mode
    out = Path(output_root) / result_id
    if (out / "summary.json").is_file() and not force:
        return json.loads((out / "summary.json").read_text())
    if mode not in {"kinematic", "ocd"}:
        raise ValueError(mode)
    if planner_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)

    token = CONFIG["modified_scenario_token"]
    scenario, cfg = build_scenario(token)
    speed_override = apply_initial_speed_override(scenario, initial_speed_mps) if initial_speed_mps else None
    expected_poses = stopped_vehicle_poses(scenario)
    original_controller = build_controller(scenario, cfg)
    if mode == "kinematic":
        controller = original_controller
        motion_model = original_controller._motion_model
    else:
        peak_scale = (
            CONFIG["ocd"]["internal_mf52_peak_scale"] if ocd_peak_scale is None else float(ocd_peak_scale)
        )
        motion_model = OCDMotionModel(
            scenario.ego_vehicle_parameters,
            physics_dt=CONFIG["ocd"]["physics_step_s"],
            base_scale=peak_scale,
        )
        controller = LoggingController(scenario, original_controller._tracker, motion_model)
    observation = build_observation(scenario)
    setup = SimulationSetup(
        time_controller=StepSimulationTimeController(scenario),
        observations=observation,
        ego_controller=controller,
        scenario=scenario,
    )
    simulation = FixedHorizonSimulation(
        setup,
        simulation_history_buffer_duration=2.0,
        fixed_horizon_steps=CONFIG["fixed_horizon_steps"],
    )
    planner = build_planner(
        expected_poses,
        device=planner_device,
        config_path=planner_config_path or INFERENCE_CONFIG,
    )
    planner.initialize(simulation.initialize())

    references, planning_times = [], []
    started = time.perf_counter()
    while simulation.is_simulation_running():
        planner_input = simulation.get_planner_input()
        call_started = time.perf_counter()
        trajectory = planner.compute_trajectory(planner_input)
        planning_times.append(time.perf_counter() - call_started)
        references.append(trajectory_array(trajectory))
        simulation.propagate(trajectory)
    wall_seconds = time.perf_counter() - started

    out.mkdir(parents=True, exist_ok=True)
    samples = simulation.history.data
    t0 = samples[0].iteration.time_point.time_us
    rows, obstacle_collision_times, any_collision_times, traffic_frames = [], [], [], []
    obstacle_tracks = [[] for _ in expected_poses]
    for sample in samples:
        time_s = (sample.iteration.time_point.time_us - t0) * 1e-6
        objects = sample.observation.tracked_objects.tracked_objects
        traffic_frames.append(
            {
                "iteration": sample.iteration.index,
                "objects": [
                    {
                        "token": obj.token,
                        "type": obj.tracked_object_type.name,
                        "x": obj.center.x,
                        "y": obj.center.y,
                        "heading": obj.center.heading,
                        "length": obj.box.length,
                        "width": obj.box.width,
                        "height": obj.box.height,
                        "speed": obj.velocity.magnitude() if hasattr(obj, "velocity") else 0.0,
                    }
                    for obj in objects
                ],
            }
        )
        stopped_now = []
        for expected in expected_poses:
            candidates = [obj for obj in objects if hasattr(obj, "velocity") and obj.velocity.magnitude() < 0.05]
            obj = min(
                candidates,
                key=lambda item: math.hypot(item.center.x - expected.x, item.center.y - expected.y),
                default=None,
            )
            stopped_now.append(obj)
        obstacle_hit = any(
            obj is not None and sample.ego_state.car_footprint.geometry.intersects(obj.box.geometry)
            for obj in stopped_now
        )
        any_hit = any(sample.ego_state.car_footprint.geometry.intersects(obj.box.geometry) for obj in objects)
        if obstacle_hit:
            obstacle_collision_times.append(time_s)
        if any_hit:
            any_collision_times.append(time_s)
        for index, obj in enumerate(stopped_now):
            if obj is not None:
                obstacle_tracks[index].append(
                    [time_s, obj.center.x, obj.center.y, obj.center.heading, obj.velocity.magnitude()]
                )
        ego = sample.ego_state
        rows.append(
            {
                "iteration": sample.iteration.index,
                "timestamp_us": sample.iteration.time_point.time_us,
                "time_s": time_s,
                "rear_axle_x_m": ego.rear_axle.x,
                "rear_axle_y_m": ego.rear_axle.y,
                "heading_rad": ego.rear_axle.heading,
                "speed_mps": ego.dynamic_car_state.speed,
                "rear_axle_vx_mps": ego.dynamic_car_state.rear_axle_velocity_2d.x,
                "rear_axle_vy_mps": ego.dynamic_car_state.rear_axle_velocity_2d.y,
                "yaw_rate_radps": ego.dynamic_car_state.angular_velocity,
                "tire_steering_angle_rad": ego.tire_steering_angle,
                "obstacle_collision": int(obstacle_hit),
                "any_collision": int(any_hit),
            }
        )
    save_csv(out / "ego_states.csv", rows)
    save_csv(out / "lqr_controls.csv", controller.control_log)
    if mode == "ocd":
        save_csv(out / "ocd_states_controls.csv", motion_model.log)
    np.savez_compressed(
        out / "reference_trajectories_global.npz",
        trajectories=np.stack(references),
        columns=np.asarray(["timestamp_us", "rear_axle_x_m", "rear_axle_y_m", "heading_rad", "speed_mps"]),
    )
    np.savez_compressed(
        out / "stopped_vehicle_tracks.npz",
        **{f"vehicle_{i}": np.asarray(track) for i, track in enumerate(obstacle_tracks)},
    )
    (out / "traffic_tracks.json").write_text(json.dumps(traffic_frames) + "\n")
    (out / "model_input_audit.json").write_text(json.dumps(planner.input_audit, indent=2) + "\n")

    vehicle = scenario.ego_vehicle_parameters
    summary = {
        "result_id": result_id,
        "mode": mode,
        "modified_token": token,
        "scenario_type": scenario.scenario_type,
        "map_name": scenario.map_api.map_name,
        "log_name": scenario.log_name,
        "initial_speed_mps": scenario.initial_ego_state.dynamic_car_state.speed,
        "initial_speed_override": speed_override,
        "initial_steering_angle_rad": scenario.initial_ego_state.tire_steering_angle,
        "frames": len(rows),
        "duration_seconds": rows[-1]["time_s"],
        "planner_calls": len(references),
        "mean_planner_seconds": float(np.mean(planning_times)),
        "wall_seconds": wall_seconds,
        "minimum_speed_mps": min(row["speed_mps"] for row in rows),
        "final_speed_mps": rows[-1]["speed_mps"],
        "maximum_abs_lateral_velocity_mps": max(abs(row["rear_axle_vy_mps"]) for row in rows),
        "maximum_abs_yaw_rate_radps": max(abs(row["yaw_rate_radps"]) for row in rows),
        "first_obstacle_collision_time_s": min(obstacle_collision_times) if obstacle_collision_times else None,
        "first_any_collision_time_s": min(any_collision_times) if any_collision_times else None,
        "stopped_vehicle_poses": [[pose.x, pose.y, pose.heading] for pose in expected_poses],
        "stopped_vehicle_seen_all_calls": all(
            all(item["slot"] is not None for item in audit["stopped_vehicles"]) for audit in planner.input_audit
        ),
        "fixed_horizon_steps": CONFIG["fixed_horizon_steps"],
        "seed": 0,
        "cfg_weight": 1.0,
        "sample_steps": 4,
        "planner_device": planner_device,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "traffic": "interPlan IDMAgents",
        "tracker": "official nuPlan LQR, unchanged",
        "motion_model": "nuPlan kinematic bicycle" if mode == "kinematic" else "Open Car Dynamics double track MF52",
        "nuplan_collision_footprint": {
            "vehicle": vehicle.vehicle_name,
            "length_m": vehicle.length,
            "width_m": vehicle.width,
            "wheelbase_m": vehicle.wheel_base,
            "rear_axle_to_geometric_center_m": vehicle.rear_axle_to_center,
            "rear_axle_to_cog_m": vehicle.cog_position_from_rear_axle,
        },
        "ocd_vehicle": ocd_vehicle_metadata(peak_scale) if mode == "ocd" else None,
        "ocd_upstream_commit": motion_model.backend.commit if mode == "ocd" else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["kinematic", "ocd", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    modes = ["kinematic", "ocd"] if args.mode == "both" else [args.mode]
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = [run(mode, args.force) for mode in modes]
    (RESULTS / "batch_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
