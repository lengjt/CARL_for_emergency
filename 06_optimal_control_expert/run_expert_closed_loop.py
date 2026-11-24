#!/usr/bin/env python3
"""Run optimal-control references through the unchanged nuPlan LQR and OCD."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from shapely.geometry import Point, Polygon

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
STAGE4 = RESEARCH / "04_nuplan_static_obstacle"
STAGE5 = RESEARCH / "05_nuplan_ocd_comparison"
sys.path[:0] = [str(STAGE4), str(STAGE5), str(HERE)]

from run_official_interplan import (  # noqa: E402
    IDMAgents, LoggingController, SimulationSetup, StepSimulationTimeController,
    build_controller, build_scenario, save_csv, stopped_vehicle_poses, trajectory_array,
)
from run_comparison import (  # noqa: E402
    CONFIG as STAGE5_CONFIG, FixedHorizonSimulation, apply_initial_speed_override,
    build_observation, ocd_vehicle_metadata,
)
from ocd_motion_model import OCDMotionModel  # noqa: E402
from nuplan.common.actor_state.ego_state import EgoState  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint  # noqa: E402
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory  # noqa: E402
from expert_optimizer import ExpertConfig, OptimalControlExpert, SceneGeometry  # noqa: E402


class ExpertTrajectoryPlanner:
    """Replan at a configurable period and return an 8 s nuPlan reference every 0.1 s."""

    def __init__(self, geometry, config, replan_period_s):
        self.geometry = geometry
        self.config = config
        self.replan_period_s = float(replan_period_s)
        self.optimizer = OptimalControlExpert(geometry, config, pass_side=-1)
        self.cached = None
        self.cached_timestamp_us = None
        self.last_attempt_timestamp_us = None
        self.solve_log = []
        self.call_log = []
        self.successful_plans = []
        self.suppressed_replans = 0

    def initialize(self, initialization):
        return None

    def _emergency_plan(self, ego, timestamp_us):
        times = np.arange(91, dtype=float) * self.config.output_dt_s
        speed0 = ego.dynamic_car_state.speed
        deceleration = -4.0
        speeds = np.maximum(0.0, speed0 + deceleration * times)
        distance = speed0 * times + 0.5 * deceleration * np.minimum(times, speed0 / 4.0) ** 2
        distance[times > speed0 / 4.0] = speed0 ** 2 / 8.0
        local0 = self.geometry.initial_vector(ego)
        x = local0[0] + distance * math.cos(local0[2])
        y = local0[1] + distance * math.sin(local0[2])
        self.cached = np.column_stack((times, x, y, np.full_like(times, local0[2]), speeds,
                                       np.full_like(times, local0[4]), np.where(speeds > 0, deceleration, 0.0)))
        self.cached_timestamp_us = timestamp_us

    def _solve_if_due(self, ego, timestamp_us):
        if self.cached is not None and self.config.maneuver == "safe_stop":
            local = self.geometry.initial_vector(ego)
            center_x = local[0] + self.config.rear_axle_to_center_m * math.cos(local[2])
            if center_x >= self.optimizer._stop_line(local[3]) - 1.0:
                self.suppressed_replans += 1
                return False
        elapsed = None if self.last_attempt_timestamp_us is None else (timestamp_us - self.last_attempt_timestamp_us) * 1e-6
        timestamp_tolerance_s = max(5e-4, 0.01 * self.replan_period_s)
        due = elapsed is None or elapsed + timestamp_tolerance_s >= self.replan_period_s
        if not due:
            return False
        initial_vector = self.geometry.initial_vector(ego)
        result = self.optimizer.solve(
            initial_vector,
            warm_shift_s=self.replan_period_s if self.cached is not None else None,
        )
        warm_status = result.status
        cold_retry = False
        if not result.success and self.optimizer.previous_states is not None:
            cold_retry = True
            first_time, first_iterations, first_message = result.solve_time_s, result.iterations, result.message
            self.optimizer.previous_states = None
            self.optimizer.previous_controls = None
            result = self.optimizer.solve(initial_vector)
            result.solve_time_s += first_time
            result.iterations += first_iterations
            if not result.success:
                result.message = first_message + "\nCold retry:\n" + result.message
        record = {
            "timestamp_us": timestamp_us,
            "success": result.success,
            "status": result.status,
            "solve_time_s": result.solve_time_s,
            "iterations": result.iterations,
            "objective": result.objective,
            "message": result.message,
            "warm_start_status": warm_status,
            "cold_retry": cold_retry,
            "used_previous_reference_after_failure": bool(not result.success and self.cached is not None),
        }
        self.solve_log.append(record)
        self.last_attempt_timestamp_us = timestamp_us
        if result.success:
            self.cached = result.sampled_states
            self.cached_timestamp_us = timestamp_us
            self.successful_plans.append(result.sampled_states.copy())
        elif self.cached is None:
            self._emergency_plan(ego, timestamp_us)
        return True

    def _state_at(self, relative_time_s):
        plan = self.cached
        relative_time_s = float(np.clip(relative_time_s, plan[0, 0], plan[-1, 0]))
        values = np.empty(6, dtype=float)
        for output_index, source_index in enumerate((1, 2, 3, 4, 5, 6)):
            source = np.unwrap(plan[:, source_index]) if source_index == 3 else plan[:, source_index]
            values[output_index] = np.interp(relative_time_s, plan[:, 0], source)
        return values

    def compute_trajectory(self, planner_input):
        ego = planner_input.history.current_state[0]
        timestamp_us = planner_input.iteration.time_point.time_us
        replanned = self._solve_if_due(ego, timestamp_us)
        age_s = (timestamp_us - self.cached_timestamp_us) * 1e-6
        states = []
        for index in range(81):
            offset_s = index * 0.1
            x, y, heading_local, speed, steering, acceleration = self._state_at(age_s + offset_s)
            world = self.geometry.local_to_world(np.asarray([[x, y]]))[0]
            heading = self.geometry.heading + heading_local
            yaw_rate = speed * math.tan(steering) / self.config.wheelbase_m
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(world[0], world[1], heading),
                rear_axle_velocity_2d=StateVector2D(speed, 0.0),
                rear_axle_acceleration_2d=StateVector2D(acceleration, speed * yaw_rate),
                tire_steering_angle=steering,
                time_point=TimePoint(timestamp_us + int(round(offset_s * 1e6))),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
                is_in_auto_mode=True,
                angular_vel=yaw_rate,
                angular_accel=0.0,
                tire_steering_rate=0.0,
            ))
        self.call_log.append({
            "iteration": planner_input.iteration.index,
            "timestamp_us": timestamp_us,
            "replanned": replanned,
            "reference_age_s": age_s,
            "latest_solve_success": self.solve_log[-1]["success"],
        })
        return InterpolatedTrajectory(states)


def _local_footprint(geometry, footprint):
    coordinates = np.asarray(footprint.geometry.exterior.coords, dtype=float)
    return Polygon(geometry.world_to_local(coordinates))


def run(
    period_s: float,
    force: bool = False,
    output_root: Path | None = None,
    output_name: str | None = None,
    initial_speed_mps: float = 15.0,
    peak_scale: float = 1.0,
    maneuver: str = "safe_stop",
):
    result_id = output_name or f"replan_{str(period_s).replace('.', 'p')}s"
    root = output_root or HERE / "results"
    out = root / result_id
    if (out / "summary.json").is_file() and not force:
        return json.loads((out / "summary.json").read_text())

    scenario, cfg = build_scenario("81f400b414215325-s0")
    apply_initial_speed_override(scenario, initial_speed_mps)
    obstacle_poses = stopped_vehicle_poses(scenario)
    geometry = SceneGeometry(scenario, obstacle_poses[0])
    expert_config = ExpertConfig(capability_scale=peak_scale, maneuver=maneuver)
    planner = ExpertTrajectoryPlanner(geometry, expert_config, period_s)

    original_controller = build_controller(scenario, cfg)
    motion_model = OCDMotionModel(scenario.ego_vehicle_parameters, physics_dt=0.001, base_scale=peak_scale)
    controller = LoggingController(scenario, original_controller._tracker, motion_model)
    observation = build_observation(scenario)
    setup = SimulationSetup(
        time_controller=StepSimulationTimeController(scenario), observations=observation,
        ego_controller=controller, scenario=scenario,
    )
    simulation = FixedHorizonSimulation(
        setup, simulation_history_buffer_duration=2.0,
        fixed_horizon_steps=STAGE5_CONFIG["fixed_horizon_steps"],
    )
    planner.initialize(simulation.initialize())

    references = []
    started = time.perf_counter()
    while simulation.is_simulation_running():
        planner_input = simulation.get_planner_input()
        trajectory = planner.compute_trajectory(planner_input)
        references.append(trajectory_array(trajectory))
        simulation.propagate(trajectory)
    wall_time_s = time.perf_counter() - started

    out.mkdir(parents=True, exist_ok=True)
    samples = simulation.history.data
    t0 = samples[0].iteration.time_point.time_us
    rows, traffic_frames, obstacle_tracks = [], [], [[] for _ in obstacle_poses]
    collision_times, offroad_times = [], []
    for sample in samples:
        time_s = (sample.iteration.time_point.time_us - t0) * 1e-6
        objects = sample.observation.tracked_objects.tracked_objects
        traffic_frames.append({
            "iteration": sample.iteration.index,
            "objects": [{
                "token": item.token, "type": item.tracked_object_type.name,
                "x": item.center.x, "y": item.center.y, "heading": item.center.heading,
                "length": item.box.length, "width": item.box.width, "height": item.box.height,
                "speed": item.velocity.magnitude() if hasattr(item, "velocity") else 0.0,
            } for item in objects],
        })
        stopped = []
        for pose in obstacle_poses:
            candidates = [item for item in objects if hasattr(item, "velocity") and item.velocity.magnitude() < 0.05]
            item = min(candidates, key=lambda obj: math.hypot(obj.center.x - pose.x, obj.center.y - pose.y), default=None)
            stopped.append(item)
        collision = any(item is not None and sample.ego_state.car_footprint.geometry.intersects(item.box.geometry)
                        for item in stopped)
        if collision:
            collision_times.append(time_s)
        footprint = _local_footprint(geometry, sample.ego_state.car_footprint)
        on_route = geometry.route_polygon_local.buffer(1e-6).covers(footprint)
        if on_route:
            road_margin = footprint.distance(geometry.route_polygon_local.boundary)
        else:
            road_margin = -max(
                geometry.route_polygon_local.distance(Point(point)) for point in footprint.exterior.coords
            )
        if not on_route:
            offroad_times.append(time_s)
        for index, item in enumerate(stopped):
            if item is not None:
                obstacle_tracks[index].append([time_s, item.center.x, item.center.y, item.center.heading, item.velocity.magnitude()])
        ego = sample.ego_state
        rows.append({
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
            "road_margin_m": road_margin,
            "off_route_road": int(not on_route),
            "obstacle_collision": int(collision),
        })

    save_csv(out / "ego_states.csv", rows)
    save_csv(out / "lqr_controls.csv", controller.control_log)
    save_csv(out / "ocd_states_controls.csv", motion_model.log)
    save_csv(out / "expert_solve_log.csv", planner.solve_log)
    save_csv(out / "planner_call_log.csv", planner.call_log)
    np.savez_compressed(out / "reference_trajectories_global.npz", trajectories=np.stack(references),
                        columns=np.asarray(["timestamp_us", "rear_axle_x_m", "rear_axle_y_m", "heading_rad", "speed_mps"]))
    np.savez_compressed(out / "expert_optimized_trajectories_local.npz",
                        trajectories=np.stack(planner.successful_plans) if planner.successful_plans else np.empty((0, 91, 7)),
                        columns=np.asarray(["relative_time_s", "rear_axle_x_m", "rear_axle_y_m", "heading_rad",
                                            "speed_mps", "steering_rad", "acceleration_mps2"]))
    np.savez_compressed(out / "stopped_vehicle_tracks.npz",
                        **{f"vehicle_{i}": np.asarray(track) for i, track in enumerate(obstacle_tracks)})
    (out / "traffic_tracks.json").write_text(json.dumps(traffic_frames) + "\n")

    solve_times = np.asarray([item["solve_time_s"] for item in planner.solve_log], dtype=float)
    successes = sum(item["success"] for item in planner.solve_log)
    local_actual = geometry.world_to_local(np.asarray([
        [item["rear_axle_x_m"], item["rear_axle_y_m"]] for item in rows
    ]))
    pass_threshold_m = geometry.obstacle_x + 0.5 * (
        expert_config.vehicle_length_m + geometry.obstacle_length
    ) + 5.0
    passed_obstacle = bool(np.max(local_actual[:, 0]) >= pass_threshold_m)
    summary = {
        "result_id": result_id,
        "experiment": (
            f"真实 nuPlan 场景，自车初速度设为 {initial_speed_mps:g} m/s，"
            f"OCD 内部 MF5.2 峰值缩放设为 {peak_scale:g}；速度和缩放均为实验设定。"
        ),
        "planner": "constrained optimal-control expert",
        "reference_interface": "time-parameterized nuPlan InterpolatedTrajectory",
        "execution_chain": "expert reference -> unchanged nuPlan LQR -> unchanged adapter -> OCD",
        "direct_control_to_ocd": False,
        "replan_period_s": period_s,
        "lqr_ocd_period_s": 0.1,
        "initial_speed_mps": initial_speed_mps,
        "internal_peak_scale": peak_scale,
        "maneuver": maneuver,
        "frames": len(rows),
        "duration_s": rows[-1]["time_s"],
        "first_collision_time_s": min(collision_times) if collision_times else None,
        "first_offroad_time_s": min(offroad_times) if offroad_times else None,
        "passed_obstacle": passed_obstacle,
        "pass_threshold_local_x_m": pass_threshold_m,
        "maximum_actual_local_x_m": float(np.max(local_actual[:, 0])),
        "minimum_road_margin_m": min(item["road_margin_m"] for item in rows),
        "minimum_speed_mps": min(item["speed_mps"] for item in rows),
        "final_speed_mps": rows[-1]["speed_mps"],
        "solve_attempts": len(planner.solve_log),
        "solve_successes": successes,
        "solve_failures": len(planner.solve_log) - successes,
        "suppressed_replans_after_stop_commitment": planner.suppressed_replans,
        "solve_time_mean_s": float(solve_times.mean()),
        "solve_time_p95_s": float(np.percentile(solve_times, 95)),
        "solve_time_max_s": float(solve_times.max()),
        "deadline_misses": int(np.sum(solve_times > period_s)),
        "deadline_miss_rate": float(np.mean(solve_times > period_s)),
        "wall_time_s": wall_time_s,
        "expert": planner.optimizer.metadata(),
        "tracker": "official nuPlan LQR, unchanged",
        "motion_model": "Open Car Dynamics double track MF52",
        "ocd_vehicle": ocd_vehicle_metadata(peak_scale),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", type=float, choices=[0.1, 0.5, 1.0])
    parser.add_argument("--speed", type=float, default=15.0)
    parser.add_argument("--peak-scale", type=float, default=1.0)
    parser.add_argument("--maneuver", choices=["safe_stop", "pass"], default="safe_stop")
    parser.add_argument("--output-name")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    periods = [args.period] if args.period else [0.1, 0.5, 1.0]
    summaries = [run(
        period, args.force, args.output_root, args.output_name,
        args.speed, args.peak_scale, args.maneuver,
    ) for period in periods]
    results = args.output_root or HERE / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "replan_period_summaries.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
