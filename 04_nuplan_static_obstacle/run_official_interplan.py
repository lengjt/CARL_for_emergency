#!/usr/bin/env python3
"""Run released Flow Planner on official interPlan special scenarios with resumable logs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
FLOW_ROOT = RESEARCH / "01_flow_planner"
INTERPLAN_ROOT = HERE / "upstream/interPlan"
sys.path[:0] = [
    str(INTERPLAN_ROOT),
    str(FLOW_ROOT / "upstream/Flow-Planner"),
    str(FLOW_ROOT / "upstream/nuplan-devkit"),
]
os.environ.setdefault("NUPLAN_DATA_ROOT", str(FLOW_ROOT / "data"))
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(FLOW_ROOT / "data/maps"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache/matplotlib"))

from flow_planner.planner import FlowPlanner  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from interplan.planning.scenario_builder.scenario_utils import (  # noqa: E402
    ModificationsSerializableDictionary as ModDict,
)
from interplan.planning.simulation.interplan_simulation import Simulation  # noqa: E402
from interplan.planning.simulation.observation.idm_modified_agents import IDMAgents  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2  # noqa: E402
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel  # noqa: E402
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker  # noqa: E402
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController  # noqa: E402
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput  # noqa: E402
from nuplan.planning.simulation.simulation_setup import SimulationSetup  # noqa: E402
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (  # noqa: E402
    StepSimulationTimeController,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402
from nuplan.planning.utils.multithreading.worker_sequential import Sequential  # noqa: E402

UPSTREAM_CONFIG = FLOW_ROOT / "artifacts/nuplan/exp/simulation/closed_loop_nonreactive_agents/flow_planner_mini/2026-10-13_21-31-26/code/hydra/config.yaml"
INFERENCE_CONFIG = FLOW_ROOT / "artifacts/nuplan/inference_config.yaml"
CHECKPOINT = FLOW_ROOT / "checkpoints/model.pth"
INTERPLAN10 = INTERPLAN_ROOT / "interplan/planning/script/config/common/scenario_filter/interplan10.yaml"
MODIFICATIONS = INTERPLAN_ROOT / "interplan/planning/script/config/common/scenario_filter/modifications/interPlan_modifications.yaml"
DATA_DIR = FLOW_ROOT / "data/nuplan-v1.1/splits/test"
RESULTS = HERE / "results_official_interplan"
SELECTED_TYPES = {
    "nudge_around_parked_vehicle",
    "overtake_parked_vehicle",
    "avoid_crashed_cars",
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_type(token: str, modifications: Dict[str, Any]) -> str | None:
    base, suffix = token.rsplit("-", 1)
    if not suffix.startswith("s"):
        return None
    number = suffix[1:]
    details = modifications["modification_details_dictionary"][base]
    return details.get("special_scenario", {}).get(number, {}).get("type")


def selected_tokens(categories: List[str]) -> List[str]:
    split = OmegaConf.to_container(OmegaConf.load(INTERPLAN10), resolve=False)
    mods = OmegaConf.to_container(OmegaConf.load(MODIFICATIONS), resolve=False)
    return [
        token
        for token in split["scenario_tokens"]
        if token_type(token, mods) in categories
    ]


def build_scenario(
    modified_token: str,
    stopped_vehicle_override: List[List[float]] | None = None,
):
    base_cfg = OmegaConf.load(UPSTREAM_CONFIG)
    OmegaConf.set_struct(base_cfg, False)
    base_cfg.scenario_builder._target_ = (
        "interplan.planning.scenario_builder.nuplan_db.modified_nuplan_scenario_builder."
        "NuPlanModifiedScenarioBuilder"
    )
    base_cfg.scenario_builder.data_root = str(DATA_DIR)
    base_cfg.scenario_builder.map_root = str(FLOW_ROOT / "data/maps")
    base_cfg.scenario_builder.sensor_root = str(FLOW_ROOT / "data/nuplan-v1.1/sensor_blobs")
    base_cfg.scenario_builder.db_files = [str(path) for path in sorted(DATA_DIR.glob("*.db"))]
    base_cfg.scenario_builder.max_workers = 1
    base_cfg.scenario_builder.verbose = False
    base_cfg.scenario_filter._target_ = (
        "interplan.planning.scenario_builder.benchmark_scenario_filter.BenchmarkScenarioFilter"
    )
    mods = OmegaConf.load(MODIFICATIONS)
    mods.accel_max = 1
    mods.decel_max = 2
    base, suffix = modified_token.rsplit("-", 1)
    if stopped_vehicle_override is not None:
        number = suffix[1:]
        mods.modification_details_dictionary[base].special_scenario[number].stopped_vehicle = [
            [float(value) for value in pose] for pose in stopped_vehicle_override
        ]
    mods.scenario_specifics = [[suffix]]
    base_cfg.scenario_filter.modifications = mods
    base_cfg.scenario_filter.scenario_tokens = [base]
    base_cfg.scenario_filter.log_names = None
    base_cfg.scenario_filter.limit_total_scenarios = 1
    base_cfg.scenario_filter.num_scenarios_per_type = None
    base_cfg.scenario_filter.shuffle = False
    base_cfg.scenario_filter.remove_invalid_goals = False
    builder = instantiate(base_cfg.scenario_builder)
    scenarios = builder.get_scenarios(instantiate(base_cfg.scenario_filter), Sequential())
    if len(scenarios) != 1:
        raise RuntimeError(f"{modified_token}: expected one scenario, got {len(scenarios)}")
    scenario = scenarios[0]
    expected_type = token_type(modified_token, OmegaConf.to_container(mods, resolve=False))
    if scenario.token != base or scenario.scenario_type != expected_type:
        raise RuntimeError(
            f"Wrong scenario for {modified_token}: {scenario.token} / {scenario.scenario_type}"
        )
    return scenario, base_cfg


def stopped_vehicle_poses(scenario) -> List[StateSE2]:
    number = str(scenario.modification["special_scenario"])
    records = scenario.mod_details["special_scenario"][number].get("stopped_vehicle", [])
    return [StateSE2(float(x), float(y), float(h)) for x, y, h in records]


class LoggingController(TwoStageController):
    def __init__(self, scenario, tracker, motion_model):
        super().__init__(scenario, tracker, motion_model)
        self.control_log = []

    def reset(self):
        super().reset()
        self.control_log = []

    def update_state(self, current_iteration, next_iteration, ego_state, trajectory):
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


def build_controller(scenario, cfg):
    lqr = cfg.ego_controller.tracker
    tracker = LQRTracker(
        q_longitudinal=np.asarray(lqr.q_longitudinal, dtype=np.float64),
        r_longitudinal=np.asarray(lqr.r_longitudinal, dtype=np.float64),
        q_lateral=np.asarray(lqr.q_lateral, dtype=np.float64),
        r_lateral=np.asarray(lqr.r_lateral, dtype=np.float64),
        discretization_time=float(lqr.discretization_time),
        tracking_horizon=int(lqr.tracking_horizon),
        jerk_penalty=float(lqr.jerk_penalty),
        curvature_rate_penalty=float(lqr.curvature_rate_penalty),
        stopping_proportional_gain=float(lqr.stopping_proportional_gain),
        stopping_velocity=float(lqr.stopping_velocity),
        vehicle=scenario.ego_vehicle_parameters,
    )
    model = cfg.ego_controller.motion_model
    motion_model = KinematicBicycleModel(
        vehicle=scenario.ego_vehicle_parameters,
        max_steering_angle=float(model.get("max_steering_angle", np.pi / 3)),
        accel_time_constant=float(model.get("accel_time_constant", 0.2)),
        steering_angle_time_constant=float(model.get("steering_angle_time_constant", 0.05)),
    )
    return LoggingController(scenario, tracker, motion_model)


class OfficialAuditedFlowPlanner(FlowPlanner):
    def __init__(self, *args, expected_poses: List[StateSE2], **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_poses = expected_poses
        self.input_audit = []

    def compute_planner_trajectory(self, current_input: PlannerInput):
        inputs = self.planner_input_to_model_inputs(current_input)
        agents = inputs.neighbor_past.detach().cpu().numpy()
        if agents.ndim == 4:
            agents = agents[0]
        current = agents[:, -1, :]
        ego = current_input.history.current_state[0].rear_axle
        c, s = math.cos(ego.heading), math.sin(ego.heading)
        records = []
        for pose in self.expected_poses:
            dx, dy = pose.x - ego.x, pose.y - ego.y
            expected = np.asarray([c * dx + s * dy, -s * dx + c * dy])
            vehicle_slots = np.flatnonzero(current[:, 8] > 0.5)
            slot = None
            error = None
            if len(vehicle_slots):
                errors = np.linalg.norm(current[vehicle_slots, :2] - expected, axis=1)
                best = int(np.argmin(errors))
                error = float(errors[best])
                if error < 0.25:
                    slot = int(vehicle_slots[best])
            records.append(
                {
                    "global_pose": [pose.x, pose.y, pose.heading],
                    "slot": slot,
                    "position_error_m": error,
                    "feature": current[slot].tolist() if slot is not None else None,
                }
            )
        self.input_audit.append(
            {
                "iteration": current_input.iteration.index,
                "timestamp_us": current_input.iteration.time_point.time_us,
                "nonzero_agent_slots": int(np.sum(np.any(np.abs(current) > 1e-7, axis=1))),
                "stopped_vehicles": records,
            }
        )
        outputs = self.core.inference(self._planner, inputs, use_cfg=self.use_cfg, cfg_weight=self.cfg_weight)
        return InterpolatedTrajectory(self.outputs_to_trajectory(outputs, current_input.history.ego_states))


def build_planner(expected_poses, device="cuda", config_path=INFERENCE_CONFIG):
    effective_config_path = Path(config_path)
    temporary_config = None
    loaded_config = OmegaConf.load(effective_config_path)
    if str(loaded_config.get("device", "cuda")) != device:
        OmegaConf.set_struct(loaded_config, False)
        loaded_config.device = device
        loaded_config.model.device = device
        temporary_config = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        temporary_config.close()
        effective_config_path = Path(temporary_config.name)
        OmegaConf.save(loaded_config, effective_config_path)
    planner = OfficialAuditedFlowPlanner(
        config_path=str(effective_config_path),
        ckpt_path=str(CHECKPOINT),
        past_trajectory_sampling=TrajectorySampling(num_poses=20, time_horizon=2.0),
        future_trajectory_sampling=TrajectorySampling(num_poses=80, time_horizon=8.0),
        enable_ema=False,
        device=device,
        use_cfg=True,
        cfg_weight=1.0,
        expected_poses=expected_poses,
    )
    if int(planner._config.model.flow_ode.sample_steps) != 4:
        raise RuntimeError("Released inference configuration no longer uses four sampling steps")
    if temporary_config is not None:
        effective_config_path.unlink(missing_ok=True)
    return planner


def trajectory_array(trajectory):
    return np.asarray(
        [
            [
                state.time_point.time_us,
                state.rear_axle.x,
                state.rear_axle.y,
                state.rear_axle.heading,
                state.dynamic_car_state.speed,
            ]
            for state in trajectory.get_sampled_trajectory()
        ],
        dtype=np.float64,
    )


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    modified_token: str,
    force: bool = False,
    output_root: Path = RESULTS,
    output_name: str | None = None,
    stopped_vehicle_override: List[List[float]] | None = None,
    summary_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run one scenario, optionally overriding only its interPlan stopped-vehicle poses."""
    result_id = output_name or modified_token
    out = Path(output_root) / result_id
    if (out / "summary.json").is_file() and not force:
        return json.loads((out / "summary.json").read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)

    scenario, cfg = build_scenario(modified_token, stopped_vehicle_override)
    expected_poses = stopped_vehicle_poses(scenario)
    observation = IDMAgents(
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
    controller = build_controller(scenario, cfg)
    setup = SimulationSetup(
        time_controller=StepSimulationTimeController(scenario),
        observations=observation,
        ego_controller=controller,
        scenario=scenario,
    )
    simulation = Simulation(setup, simulation_history_buffer_duration=2.0)
    planner = build_planner(expected_poses)
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
    rows, obstacle_collisions, any_collisions = [], [], []
    traffic_frames = []
    obstacle_motion = [[] for _ in expected_poses]
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
            candidates = [
                obj for obj in objects
                if hasattr(obj, "velocity") and obj.velocity.magnitude() < 0.05
            ]
            obj = min(candidates, key=lambda item: math.hypot(item.center.x - expected.x, item.center.y - expected.y), default=None)
            stopped_now.append(obj)
        obstacle_hit = any(
            obj is not None and sample.ego_state.car_footprint.geometry.intersects(obj.box.geometry)
            for obj in stopped_now
        )
        any_hit = any(sample.ego_state.car_footprint.geometry.intersects(obj.box.geometry) for obj in objects)
        if obstacle_hit:
            obstacle_collisions.append(time_s)
        if any_hit:
            any_collisions.append(time_s)
        for index, obj in enumerate(stopped_now):
            if obj is not None:
                obstacle_motion[index].append([time_s, obj.center.x, obj.center.y, obj.center.heading, obj.velocity.magnitude()])
        rows.append(
            {
                "iteration": sample.iteration.index,
                "timestamp_us": sample.iteration.time_point.time_us,
                "time_s": time_s,
                "rear_axle_x_m": sample.ego_state.rear_axle.x,
                "rear_axle_y_m": sample.ego_state.rear_axle.y,
                "heading_rad": sample.ego_state.rear_axle.heading,
                "speed_mps": sample.ego_state.dynamic_car_state.speed,
                "obstacle_collision": int(obstacle_hit),
                "any_collision": int(any_hit),
            }
        )
    save_csv(out / "ego_states.csv", rows)
    save_csv(out / "lqr_controls.csv", controller.control_log)
    np.savez_compressed(
        out / "reference_trajectories_global.npz",
        trajectories=np.stack(references),
        columns=np.asarray(["timestamp_us", "rear_axle_x_m", "rear_axle_y_m", "heading_rad", "speed_mps"]),
    )
    np.savez_compressed(out / "stopped_vehicle_tracks.npz", **{f"vehicle_{i}": np.asarray(track) for i, track in enumerate(obstacle_motion)})
    (out / "traffic_tracks.json").write_text(json.dumps(traffic_frames) + "\n")
    (out / "model_input_audit.json").write_text(json.dumps(planner.input_audit, indent=2) + "\n")
    visibility = [
        all(record["slot"] is not None for record in audit["stopped_vehicles"])
        for audit in planner.input_audit
    ]
    summary = {
        "result_id": result_id,
        "modified_token": modified_token,
        "base_token": scenario.token,
        "scenario_type": scenario.scenario_type,
        "map_name": scenario.map_api.map_name,
        "log_name": scenario.log_name,
        "initial_speed_mps": scenario.initial_ego_state.dynamic_car_state.speed,
        "frames": len(samples),
        "duration_seconds": rows[-1]["time_s"],
        "planner_calls": len(references),
        "mean_planner_seconds": float(np.mean(planning_times)),
        "wall_seconds": wall_seconds,
        "minimum_speed_mps": min(row["speed_mps"] for row in rows),
        "final_speed_mps": rows[-1]["speed_mps"],
        "first_obstacle_collision_time_s": min(obstacle_collisions) if obstacle_collisions else None,
        "first_any_collision_time_s": min(any_collisions) if any_collisions else None,
        "stopped_vehicle_poses": [[p.x, p.y, p.heading] for p in expected_poses],
        "stopped_vehicles_seen_calls": int(sum(visibility)),
        "stopped_vehicles_seen_all_calls": bool(all(visibility)),
        "seed": 0,
        "cfg_weight": 1.0,
        "sample_steps": 4,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "observation": "interPlan IDMAgents",
        "tracker": "nuPlan LQR",
        "motion_model": "nuPlan kinematic bicycle",
    }
    if summary_metadata:
        summary.update(summary_metadata)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=sorted(SELECTED_TYPES), choices=sorted(SELECTED_TYPES))
    parser.add_argument("--token")
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tokens = [args.token] if args.token else selected_tokens(args.categories)
    if args.max_scenarios is not None:
        tokens = tokens[: args.max_scenarios]
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, token in enumerate(tokens, 1):
        print(f"[{index}/{len(tokens)}] {token}", flush=True)
        summaries.append(run_one(token, force=args.force))
    (RESULTS / "batch_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
