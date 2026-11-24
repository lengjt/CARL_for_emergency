#!/usr/bin/env python3
"""Rank officially extractable nuPlan-mini scenes with a straight adjacent lane."""
import json
import math
import sys
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
FLOW_ROOT = Path("/home/ljt23/research_ca/01_flow_planner")
sys.path.insert(0, str(FLOW_ROOT / "upstream" / "nuplan-devkit"))
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


def angle_error(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def main():
    source = FLOW_ROOT / "artifacts/nuplan/exp/simulation/closed_loop_nonreactive_agents/flow_planner_mini/2026-10-13_21-31-26/code/hydra/config.yaml"
    cfg = OmegaConf.load(source)
    cfg.scenario_builder.data_root = str(FLOW_ROOT / "data/nuplan-v1.1/splits/mini")
    cfg.scenario_builder.map_root = str(FLOW_ROOT / "data/maps")
    cfg.scenario_builder.sensor_root = str(FLOW_ROOT / "data/nuplan-v1.1/sensor_blobs")
    cfg.scenario_builder.db_files = None
    cfg.scenario_builder.max_workers = 1
    cfg.scenario_builder.verbose = False
    cfg.scenario_filter.limit_total_scenarios = None
    cfg.scenario_filter.remove_invalid_goals = True
    cfg.scenario_filter.shuffle = False
    builder = instantiate(cfg.scenario_builder)
    scenarios = builder.get_scenarios(instantiate(cfg.scenario_filter), Sequential())
    ranked = []
    for scenario in scenarios:
        states = [scenario.initial_ego_state] + list(scenario.get_expert_ego_trajectory())
        if scenario.initial_ego_state.dynamic_car_state.speed < 5.5:
            continue
        cumulative = [0.0]
        for previous, current in zip(states, states[1:]):
            cumulative.append(cumulative[-1] + math.hypot(
                current.rear_axle.x - previous.rear_axle.x,
                current.rear_axle.y - previous.rear_axle.y,
            ))
        if cumulative[-1] < 55:
            continue
        target = min(range(len(states)), key=lambda i: abs(cumulative[i] - 40.0))
        pose = states[target].rear_axle
        lane = scenario.map_api.get_one_map_object(Point2D(pose.x, pose.y), SemanticMapLayer.LANE)
        if lane is None or not lane.adjacent_edges:
            continue
        first60 = [i for i, d in enumerate(cumulative) if d <= 60]
        heading_range = max(angle_error(states[i].rear_axle.heading, states[0].rear_axle.heading) for i in first60)
        if heading_range > math.radians(8):
            continue
        connector_distance = None
        for i in first60:
            p = states[i].rear_axle
            connectors = scenario.map_api.get_all_map_objects(Point2D(p.x, p.y), SemanticMapLayer.LANE_CONNECTOR)
            if connectors:
                connector_distance = cumulative[i]
                break
        if connector_distance is not None:
            continue
        lane_pose = lane.baseline_path.get_nearest_pose_from_position(Point2D(pose.x, pose.y))
        best_adj = None
        for adjacent in lane.adjacent_edges:
            if adjacent is None:
                continue
            apose = adjacent.baseline_path.get_nearest_pose_from_position(Point2D(lane_pose.x, lane_pose.y))
            separation = math.hypot(apose.x - lane_pose.x, apose.y - lane_pose.y)
            heading_delta = angle_error(apose.heading, lane_pose.heading)
            if heading_delta < math.radians(15) and (best_adj is None or separation > best_adj[0]):
                best_adj = (separation, adjacent.id, adjacent.get_roadblock_id(), heading_delta)
        if best_adj is None or best_adj[0] < 3.0:
            continue
        initial = scenario.initial_ego_state.rear_axle
        obs = scenario.get_tracked_objects_at_iteration(0)
        closer_statics = sum(
            math.hypot(obj.center.x - initial.x, obj.center.y - initial.y) < 40
            for obj in obs.tracked_objects.tracked_objects
            if obj.tracked_object_type.name not in ("VEHICLE", "PEDESTRIAN", "BICYCLE")
        )
        ranked.append({
            "token": scenario.token,
            "type": scenario.scenario_type,
            "map": scenario.map_api.map_name,
            "log": scenario.log_name,
            "database": str(scenario._log_file),
            "initial_speed_mps": scenario.initial_ego_state.dynamic_car_state.speed,
            "heading_range_first_60m_deg": math.degrees(heading_range),
            "lane_id": lane.id,
            "roadblock_id": lane.get_roadblock_id(),
            "adjacent_lane_id": best_adj[1],
            "adjacent_roadblock_id": best_adj[2],
            "lane_center_separation_m": best_adj[0],
            "closer_static_count": closer_statics,
            "barrier_pose": {"center_x_m": lane_pose.x, "center_y_m": lane_pose.y, "heading_rad": lane_pose.heading},
        })
        if len(ranked) >= 12:
            break
    ranked.sort(key=lambda x: (x["closer_static_count"], x["heading_range_first_60m_deg"], -x["lane_center_separation_m"]))
    output = HERE / "wide_candidate_ranking.json"
    output.write_text(json.dumps(ranked[:30], indent=2) + "\n")
    print(json.dumps({"official_scenarios": len(scenarios), "eligible": len(ranked), "top": ranked[:15]}, indent=2))


if __name__ == "__main__":
    main()
