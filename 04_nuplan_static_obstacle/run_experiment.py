#!/usr/bin/env python3
"""Run one or both small real-data Flow Planner closed-loop experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


CONFIG_PATH = HERE / "config.yaml"
CFG = load_config(CONFIG_PATH)
FLOW_ROOT = Path(CFG["project"]["flow_planner_root"])
sys.path.insert(0, str(FLOW_ROOT / "upstream" / "Flow-Planner"))
sys.path.insert(0, str(FLOW_ROOT / "upstream" / "nuplan-devkit"))

os.environ.setdefault("NUPLAN_DATA_ROOT", CFG["project"]["data_root"])
os.environ.setdefault("NUPLAN_MAPS_ROOT", CFG["project"]["maps_root"])
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache" / "matplotlib"))

from flow_planner.planner import FlowPlanner  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from nuplan.common.actor_state.agent import Agent  # noqa: E402
from nuplan.common.actor_state.oriented_box import OrientedBox  # noqa: E402
from nuplan.common.actor_state.scene_object import SceneObjectMetadata  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D  # noqa: E402
from nuplan.common.actor_state.static_object import StaticObject  # noqa: E402
from nuplan.common.actor_state.tracked_objects import TrackedObjects  # noqa: E402
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType  # noqa: E402
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter  # noqa: E402
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel  # noqa: E402
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker  # noqa: E402
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController  # noqa: E402
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks  # noqa: E402
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation  # noqa: E402
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput  # noqa: E402
from nuplan.planning.simulation.simulation import Simulation  # noqa: E402
from nuplan.planning.simulation.simulation_setup import SimulationSetup  # noqa: E402
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (  # noqa: E402
    StepSimulationTimeController,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402
from nuplan.planning.utils.multithreading.worker_sequential import Sequential  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_scenario(config: Dict[str, Any]):
    upstream = OmegaConf.load(config["project"]["upstream_run_config"])
    upstream.scenario_builder.data_root = str(Path(config["project"]["data_root"]) / "nuplan-v1.1/splits/mini")
    upstream.scenario_builder.map_root = config["project"]["maps_root"]
    upstream.scenario_builder.sensor_root = str(Path(config["project"]["data_root"]) / "nuplan-v1.1/sensor_blobs")
    upstream.scenario_builder.db_files = [config["scenario"]["database"]]
    upstream.scenario_builder.max_workers = 1
    upstream.scenario_builder.verbose = False
    upstream.scenario_filter.scenario_tokens = [config["scenario"]["token"]]
    upstream.scenario_filter.limit_total_scenarios = 1
    upstream.scenario_filter.shuffle = False
    upstream.scenario_filter.remove_invalid_goals = bool(config["scenario"].get("remove_invalid_goals", True))
    builder = instantiate(upstream.scenario_builder)
    scenarios = builder.get_scenarios(instantiate(upstream.scenario_filter), Sequential())
    if len(scenarios) != 1:
        raise RuntimeError(f"Expected exactly one scenario, got {len(scenarios)}")
    scenario = scenarios[0]
    if scenario.token != config["scenario"]["token"]:
        raise RuntimeError(f"Loaded wrong token: {scenario.token}")
    if scenario.map_api.map_name != config["scenario"]["expected_map"]:
        raise RuntimeError(f"Loaded wrong map: {scenario.map_api.map_name}")
    if scenario.scenario_type != config["scenario"]["expected_type"]:
        raise RuntimeError(f"Loaded wrong scenario type: {scenario.scenario_type}")
    if abs(scenario.database_interval - float(config["simulation"]["dt_seconds"])) > 1e-9:
        raise RuntimeError(f"Simulation dt mismatch: {scenario.database_interval}")
    # nuPlan's sampled duration can be one database interval shorter than the nominal horizon.
    if abs(scenario.duration_s.time_s - float(config["simulation"]["duration_seconds"])) > scenario.database_interval + 1e-6:
        raise RuntimeError(f"Simulation duration mismatch: {scenario.duration_s.time_s}")
    return scenario, upstream


def obstacle_pose(initial_state, spec: Dict[str, Any]) -> StateSE2:
    if spec.get("placement", "relative_initial_pose") == "global":
        return StateSE2(float(spec["center_x_m"]), float(spec["center_y_m"]), float(spec["heading_rad"]))
    pose = initial_state.rear_axle
    c, s = math.cos(pose.heading), math.sin(pose.heading)
    longitudinal = float(spec["longitudinal_m"])
    lateral = float(spec["lateral_m"])
    return StateSE2(
        pose.x + longitudinal * c - lateral * s,
        pose.y + longitudinal * s + lateral * c,
        pose.heading + float(spec["heading_offset_rad"]),
    )


def make_obstacle(initial_state, spec: Dict[str, Any], timestamp_us: int):
    category = TrackedObjectType[spec["category"]]
    box = OrientedBox(
        obstacle_pose(initial_state, spec),
        length=float(spec["length_m"]),
        width=float(spec["width_m"]),
        height=float(spec["height_m"]),
    )
    metadata = SceneObjectMetadata(
        timestamp_us=timestamp_us,
        token=spec["token"],
        track_token=spec["token"],
        track_id=900001,
        category_name=spec["category"].lower(),
    )
    if spec.get("representation", "static_object") == "dynamic_agent":
        return Agent(
            tracked_object_type=category,
            oriented_box=box,
            velocity=StateVector2D(float(spec.get("velocity_x_mps", 0.0)), float(spec.get("velocity_y_mps", 0.0))),
            metadata=metadata,
        )
    return StaticObject(tracked_object_type=category, oriented_box=box, metadata=metadata)


class InjectedTracksObservation(TracksObservation):
    """Log replay with one optional fixed object, static or zero-speed agent."""

    def __init__(self, scenario, spec: Dict[str, Any], enabled: bool):
        super().__init__(scenario)
        self._spec = spec
        self._enabled = enabled
        self._initial = scenario.initial_ego_state

    def get_observation(self) -> DetectionsTracks:
        original = super().get_observation()
        objects = list(original.tracked_objects.tracked_objects)
        if self._enabled:
            timestamp = self.scenario.get_time_point(self.current_iteration).time_us
            objects.append(make_obstacle(self._initial, self._spec, timestamp))
        return DetectionsTracks(TrackedObjects(objects))


class LoggingTwoStageController(TwoStageController):
    def __init__(self, scenario, tracker, motion_model):
        super().__init__(scenario, tracker, motion_model)
        self.control_log: List[Dict[str, float]] = []

    def reset(self) -> None:
        super().reset()
        self.control_log = []

    def update_state(self, current_iteration, next_iteration, ego_state, trajectory) -> None:
        sampling_time = next_iteration.time_point - current_iteration.time_point
        command = self._tracker.track_trajectory(current_iteration, next_iteration, ego_state, trajectory)
        self.control_log.append(
            {
                "iteration": current_iteration.index,
                "timestamp_us": current_iteration.time_point.time_us,
                "acceleration_mps2": float(command.rear_axle_acceleration_2d.x),
                "front_steering_rate_radps": float(command.tire_steering_rate),
            }
        )
        self._current_state = self._motion_model.propagate_state(ego_state, command, sampling_time)


class AuditedFlowPlanner(FlowPlanner):
    """Unchanged official inference with records of actual object tensors."""

    def __init__(self, *args, expected_obstacle: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_audit: List[Dict[str, Any]] = []
        self._expected_obstacle = expected_obstacle

    def compute_planner_trajectory(self, current_input: PlannerInput):
        inputs = self.planner_input_to_model_inputs(current_input)
        static_tensor = inputs.map_objects.detach().cpu().numpy()
        if static_tensor.ndim == 3:
            static_tensor = static_tensor[0]
        agent_tensor = inputs.neighbor_past.detach().cpu().numpy()
        if agent_tensor.ndim == 4:
            agent_tensor = agent_tensor[0]
        audit: Dict[str, Any] = {
            "iteration": current_input.iteration.index,
            "timestamp_us": current_input.iteration.time_point.time_us,
            "static_tensor": static_tensor.tolist(),
            "nonzero_slots": np.flatnonzero(np.any(np.abs(static_tensor) > 1e-7, axis=1)).tolist(),
            "obstacle_slot": None,
            "obstacle_agent_slot": None,
        }
        if self._expected_obstacle is not None:
            if self._expected_obstacle.get("representation", "static_object") == "dynamic_agent":
                # Agent features at the current frame: rel x/y, cos/sin h, vx/vy,
                # width/length, then VEHICLE/PEDESTRIAN/BICYCLE one-hot.
                current_agents = agent_tensor[:, -1, :]
                ego = current_input.history.current_state[0].rear_axle
                fixed = obstacle_pose(current_input.history.ego_states[0], self._expected_obstacle)
                dx, dy = fixed.x - ego.x, fixed.y - ego.y
                c, s = math.cos(ego.heading), math.sin(ego.heading)
                expected_xy = np.asarray([c * dx + s * dy, -s * dx + c * dy])
                candidate_slots = np.flatnonzero(current_agents[:, 8] > 0.5)
                matching = [
                    int(i)
                    for i in candidate_slots
                    if abs(current_agents[i, 6] - self._expected_obstacle["width_m"]) < 1e-3
                    and abs(current_agents[i, 7] - self._expected_obstacle["length_m"]) < 1e-3
                    and np.linalg.norm(current_agents[i, :2] - expected_xy) < 0.1
                ]
                audit["obstacle_agent_slot"] = matching[0] if len(matching) == 1 else None
                audit["agent_current_tensor"] = current_agents.tolist()
            else:
                # Static features: rel x/y, cos/sin heading, width, length, then category one-hot.
                barrier_slots = np.flatnonzero(static_tensor[:, 7] > 0.5)
                matching = [
                    int(i)
                    for i in barrier_slots
                    if abs(static_tensor[i, 4] - self._expected_obstacle["width_m"]) < 1e-3
                    and abs(static_tensor[i, 5] - self._expected_obstacle["length_m"]) < 1e-3
                ]
                audit["obstacle_slot"] = matching[0] if len(matching) == 1 else None
        self.input_audit.append(audit)
        outputs = self.core.inference(self._planner, inputs, use_cfg=self.use_cfg, cfg_weight=self.cfg_weight)
        return InterpolatedTrajectory(self.outputs_to_trajectory(outputs, current_input.history.ego_states))


def build_planner(config: Dict[str, Any], obstacle_enabled: bool) -> AuditedFlowPlanner:
    p = config["planner"]
    planner = AuditedFlowPlanner(
        config_path=config["project"]["inference_config"],
        ckpt_path=config["project"]["checkpoint"],
        past_trajectory_sampling=TrajectorySampling(
            num_poses=int(p["past_poses"]), time_horizon=float(p["past_horizon_seconds"])
        ),
        future_trajectory_sampling=TrajectorySampling(
            num_poses=int(p["future_poses"]), time_horizon=float(p["future_horizon_seconds"])
        ),
        enable_ema=False,
        device=p["device"],
        use_cfg=True,
        cfg_weight=float(p["cfg_weight"]),
        expected_obstacle=config["obstacle"] if obstacle_enabled else None,
    )
    # Assert the released sampling setup actually reached the model instantiated by the official adapter.
    actual_steps = int(planner._config.model.flow_ode.sample_steps)
    if actual_steps != int(p["sample_steps"]):
        raise RuntimeError(f"Sampling step mismatch: model={actual_steps}, config={p['sample_steps']}")
    return planner


def build_controller(scenario, upstream_cfg):
    lqr_cfg = upstream_cfg.ego_controller.tracker
    tracker = LQRTracker(
        q_longitudinal=np.asarray(lqr_cfg.q_longitudinal, dtype=np.float64),
        r_longitudinal=np.asarray(lqr_cfg.r_longitudinal, dtype=np.float64),
        q_lateral=np.asarray(lqr_cfg.q_lateral, dtype=np.float64),
        r_lateral=np.asarray(lqr_cfg.r_lateral, dtype=np.float64),
        discretization_time=float(lqr_cfg.discretization_time),
        tracking_horizon=int(lqr_cfg.tracking_horizon),
        jerk_penalty=float(lqr_cfg.jerk_penalty),
        curvature_rate_penalty=float(lqr_cfg.curvature_rate_penalty),
        stopping_proportional_gain=float(lqr_cfg.stopping_proportional_gain),
        stopping_velocity=float(lqr_cfg.stopping_velocity),
        vehicle=scenario.ego_vehicle_parameters,
    )
    model_cfg = upstream_cfg.ego_controller.motion_model
    motion_model = KinematicBicycleModel(
        vehicle=scenario.ego_vehicle_parameters,
        max_steering_angle=float(model_cfg.get("max_steering_angle", np.pi / 3)),
        accel_time_constant=float(model_cfg.get("accel_time_constant", 0.2)),
        steering_angle_time_constant=float(model_cfg.get("steering_angle_time_constant", 0.05)),
    )
    return LoggingTwoStageController(scenario, tracker, motion_model)


def trajectory_array(trajectory) -> np.ndarray:
    states = trajectory.get_sampled_trajectory()
    return np.asarray(
        [
            [
                state.time_point.time_us,
                state.rear_axle.x,
                state.rear_axle.y,
                state.rear_axle.heading,
                state.dynamic_car_state.speed,
            ]
            for state in states
        ],
        dtype=np.float64,
    )


def save_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_variant(config: Dict[str, Any], variant: str) -> Path:
    if variant not in ("baseline", "obstacle"):
        raise ValueError(variant)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run from the GPU-enabled host environment.")
    seed = int(config["planner"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)

    scenario, upstream_cfg = build_scenario(config)
    obstacle_enabled = variant == config["obstacle"]["enabled_variant"]
    observation = InjectedTracksObservation(scenario, config["obstacle"], obstacle_enabled)
    controller = build_controller(scenario, upstream_cfg)
    setup = SimulationSetup(
        time_controller=StepSimulationTimeController(scenario),
        observations=observation,
        ego_controller=controller,
        scenario=scenario,
    )
    simulation = Simulation(
        setup,
        simulation_history_buffer_duration=float(config["scenario"]["simulation_history_seconds"]),
    )
    planner = build_planner(config, obstacle_enabled)
    planner.initialize(simulation.initialize())

    references: List[np.ndarray] = []
    planning_seconds: List[float] = []
    coordinate_errors: List[float] = []
    started = time.perf_counter()
    while simulation.is_simulation_running():
        planner_input = simulation.get_planner_input()
        call_start = time.perf_counter()
        trajectory = planner.compute_trajectory(planner_input)
        planning_seconds.append(time.perf_counter() - call_start)
        references.append(trajectory_array(trajectory))
        current = planner_input.history.current_state[0]
        at_now = trajectory.get_state_at_time(planner_input.iteration.time_point)
        coordinate_errors.append(math.hypot(at_now.rear_axle.x - current.rear_axle.x, at_now.rear_axle.y - current.rear_axle.y))
        simulation.propagate(trajectory)
    elapsed = time.perf_counter() - started

    results_root = Path(config["project"].get("results_dir", HERE / "results"))
    out = results_root / variant
    out.mkdir(parents=True, exist_ok=True)
    samples = simulation.history.data
    t0 = samples[0].iteration.time_point.time_us
    fixed_obstacle = make_obstacle(scenario.initial_ego_state, config["obstacle"], t0)
    state_rows: List[Dict[str, Any]] = []
    obstacle_collision_times: List[float] = []
    any_collision_times: List[float] = []
    for sample in samples:
        rel_time = (sample.iteration.time_point.time_us - t0) * 1e-6
        obstacle_collision = obstacle_enabled and sample.ego_state.car_footprint.geometry.intersects(fixed_obstacle.box.geometry)
        other_collision = any(
            obj.token != config["obstacle"]["token"]
            and sample.ego_state.car_footprint.geometry.intersects(obj.box.geometry)
            for obj in sample.observation.tracked_objects.tracked_objects
        )
        if obstacle_collision:
            obstacle_collision_times.append(rel_time)
        if obstacle_collision or other_collision:
            any_collision_times.append(rel_time)
        state_rows.append(
            {
                "iteration": sample.iteration.index,
                "timestamp_us": sample.iteration.time_point.time_us,
                "time_s": rel_time,
                "rear_axle_x_m": sample.ego_state.rear_axle.x,
                "rear_axle_y_m": sample.ego_state.rear_axle.y,
                "heading_rad": sample.ego_state.rear_axle.heading,
                "speed_mps": sample.ego_state.dynamic_car_state.speed,
                "tire_steering_angle_rad": sample.ego_state.tire_steering_angle,
                "obstacle_collision": int(obstacle_collision),
                "any_collision": int(obstacle_collision or other_collision),
            }
        )
    save_csv(out / "ego_states.csv", list(state_rows[0]), state_rows)
    save_csv(
        out / "lqr_controls.csv",
        ["iteration", "timestamp_us", "acceleration_mps2", "front_steering_rate_radps"],
        controller.control_log,
    )
    reference_stack = np.stack(references)
    np.savez_compressed(
        out / "reference_trajectories_global.npz",
        trajectories=reference_stack,
        columns=np.asarray(["timestamp_us", "rear_axle_x_m", "rear_axle_y_m", "heading_rad", "speed_mps"]),
        planning_iteration=np.arange(len(references), dtype=np.int32),
        planning_timestamp_us=np.asarray([sample.iteration.time_point.time_us for sample in samples], dtype=np.int64),
    )
    obstacle_record = {
        "enabled": obstacle_enabled,
        "category": config["obstacle"]["category"],
        "token": config["obstacle"]["token"],
        "center_x_m": fixed_obstacle.center.x,
        "center_y_m": fixed_obstacle.center.y,
        "heading_rad": fixed_obstacle.center.heading,
        "length_m": fixed_obstacle.box.length,
        "width_m": fixed_obstacle.box.width,
        "height_m": fixed_obstacle.box.height,
        "representation": config["obstacle"].get("representation", "static_object"),
        "speed_mps": float(getattr(fixed_obstacle, "velocity", StateVector2D(0.0, 0.0)).magnitude()),
    }
    (out / "obstacle_geometry.json").write_text(json.dumps(obstacle_record, indent=2) + "\n")
    (out / "model_input_audit.json").write_text(json.dumps(planner.input_audit, indent=2) + "\n")
    audit_key = (
        "obstacle_agent_slot"
        if config["obstacle"].get("representation", "static_object") == "dynamic_agent"
        else "obstacle_slot"
    )
    input_seen = [a[audit_key] is not None for a in planner.input_audit] if obstacle_enabled else []
    summary = {
        "variant": variant,
        "completed": True,
        "scenario_token": scenario.token,
        "scenario_type": scenario.scenario_type,
        "map_name": scenario.map_api.map_name,
        "log_name": scenario.log_name,
        "initial_speed_mps": scenario.initial_ego_state.dynamic_car_state.speed,
        "simulation_frames": len(samples),
        "duration_seconds": state_rows[-1]["time_s"],
        "dt_seconds": scenario.database_interval,
        "seed": seed,
        "cfg_weight": config["planner"]["cfg_weight"],
        "sample_steps": config["planner"]["sample_steps"],
        "checkpoint_sha256": file_sha256(Path(config["project"]["checkpoint"])),
        "planner_calls": len(references),
        "mean_planner_seconds": float(np.mean(planning_seconds)),
        "wall_seconds": elapsed,
        "minimum_speed_mps": min(row["speed_mps"] for row in state_rows),
        "final_speed_mps": state_rows[-1]["speed_mps"],
        "obstacle": obstacle_record,
        "obstacle_seen_in_model_input_calls": int(sum(input_seen)),
        "obstacle_seen_in_all_model_input_calls": bool(all(input_seen)) if obstacle_enabled else None,
        "first_obstacle_collision_time_s": min(obstacle_collision_times) if obstacle_collision_times else None,
        "first_any_collision_time_s": min(any_collision_times) if any_collision_times else None,
        "max_reference_current_position_error_m": max(coordinate_errors),
        "lqr_control_count": len(controller.control_log),
        "traffic_mode": config["simulation"]["traffic_mode"],
        "tracker": config["simulation"]["tracker"],
        "motion_model": config["simulation"]["motion_model"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["baseline", "obstacle", "both"])
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    variants = ["baseline", "obstacle"] if args.variant == "both" else [args.variant]
    for variant in variants:
        run_variant(config, variant)


if __name__ == "__main__":
    main()
