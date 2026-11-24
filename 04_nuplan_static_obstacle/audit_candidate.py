#!/usr/bin/env python3
"""Audit a real nuPlan candidate and propose a lane-centered barrier pose."""
import argparse
import json
import math
from pathlib import Path

from shapely.geometry import Point

from run_experiment import build_scenario, load_config
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-distance", type=float, default=40.0)
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    scenario, _ = build_scenario(cfg)
    states = list(scenario.get_expert_ego_trajectory())
    states.insert(0, scenario.initial_ego_state)
    cumulative = [0.0]
    for previous, current in zip(states, states[1:]):
        cumulative.append(cumulative[-1] + distance(previous.rear_axle, current.rear_axle))
    target_index = min(range(len(states)), key=lambda i: abs(cumulative[i] - args.target_distance))
    target_state = states[target_index]
    point = Point2D(target_state.rear_axle.x, target_state.rear_axle.y)
    lane = scenario.map_api.get_one_map_object(point, SemanticMapLayer.LANE)
    if lane is None:
        raise RuntimeError("Target expert pose is not in a lane")
    adjacent = list(lane.adjacent_edges)
    lane_pose = lane.baseline_path.get_nearest_pose_from_position(point)
    adjacent_info = []
    for edge in adjacent:
        if edge is None:
            continue
        adj_pose = edge.baseline_path.get_nearest_pose_from_position(Point2D(lane_pose.x, lane_pose.y))
        heading_delta = abs((adj_pose.heading - lane_pose.heading + math.pi) % (2 * math.pi) - math.pi)
        adjacent_info.append({
            "lane_id": edge.id,
            "roadblock_id": edge.get_roadblock_id(),
            "center_separation_m": math.hypot(adj_pose.x - lane_pose.x, adj_pose.y - lane_pose.y),
            "heading_delta_rad": heading_delta,
            "polygon_clearance_for_ego_and_barrier_m": math.hypot(adj_pose.x - lane_pose.x, adj_pose.y - lane_pose.y) - (2.297 + cfg["obstacle"]["width_m"]) / 2,
        })
    connector_distances = []
    for travelled, state in zip(cumulative, states):
        objects = scenario.map_api.get_all_map_objects(
            Point2D(state.rear_axle.x, state.rear_axle.y), SemanticMapLayer.LANE_CONNECTOR
        )
        if objects:
            connector_distances.append(travelled)
    corridor = lane.polygon
    for edge in adjacent:
        if edge is None:
            continue
        corridor = corridor.union(edge.polygon)
    corridor = corridor.intersection(Point(lane_pose.x, lane_pose.y).buffer(70.0))
    moving_min_distance = {}
    early_moving_min_distance = {}
    initial_corridor_tracks = set()
    static_initial_distances = []
    initial = scenario.initial_ego_state.rear_axle
    for iteration in range(scenario.get_number_of_iterations()):
        observation = scenario.get_tracked_objects_at_iteration(iteration)
        for obj in observation.tracked_objects.tracked_objects:
            if obj.tracked_object_type.name in ("VEHICLE", "PEDESTRIAN", "BICYCLE"):
                if corridor.intersects(obj.box.geometry):
                    key = (obj.track_token, obj.tracked_object_type.name)
                    dist = math.hypot(obj.center.x - lane_pose.x, obj.center.y - lane_pose.y)
                    moving_min_distance[key] = min(dist, moving_min_distance.get(key, float("inf")))
                    if iteration <= 80:
                        early_moving_min_distance[key] = min(dist, early_moving_min_distance.get(key, float("inf")))
                    if iteration == 0:
                        initial_corridor_tracks.add(key)
            elif iteration == 0:
                static_initial_distances.append(math.hypot(obj.center.x - initial.x, obj.center.y - initial.y))
    nearest_moving = sorted(
        ({"track_token": token, "type": kind, "minimum_distance_to_barrier_m": dist}
         for (token, kind), dist in moving_min_distance.items()),
        key=lambda item: item["minimum_distance_to_barrier_m"],
    )
    nearest_early_moving = sorted(
        ({"track_token": token, "type": kind, "minimum_distance_to_barrier_m": dist}
         for (token, kind), dist in early_moving_min_distance.items()),
        key=lambda item: item["minimum_distance_to_barrier_m"],
    )
    result = {
        "token": scenario.token,
        "type": scenario.scenario_type,
        "map": scenario.map_api.map_name,
        "log": scenario.log_name,
        "duration_s": scenario.duration_s.time_s,
        "initial_speed_mps": scenario.initial_ego_state.dynamic_car_state.speed,
        "target_expert_distance_m": cumulative[target_index],
        "target_iteration": target_index,
        "barrier_pose": {"center_x_m": lane_pose.x, "center_y_m": lane_pose.y, "heading_rad": lane_pose.heading},
        "lane_id": lane.id,
        "roadblock_id": lane.get_roadblock_id(),
        "route_roadblock_ids": list(scenario.get_route_roadblock_ids()),
        "barrier_roadblock_on_route": lane.get_roadblock_id() in scenario.get_route_roadblock_ids(),
        "adjacent_lanes": adjacent_info,
        "first_lane_connector_distance_m": min(connector_distances) if connector_distances else None,
        "local_corridor_moving_track_count": len(nearest_moving),
        "initial_local_corridor_moving_track_count": len(initial_corridor_tracks),
        "first_8s_local_corridor_moving_track_count": len(nearest_early_moving),
        "nearest_first_8s_local_corridor_moving_tracks": nearest_early_moving[:10],
        "nearest_local_corridor_moving_tracks": nearest_moving[:20],
        "existing_static_count": len(static_initial_distances),
        "existing_static_closer_than_barrier_count": sum(d < cumulative[target_index] for d in static_initial_distances),
        "selection_reason": "Straight three-lane same-direction road, no lane connector on the logged path, and both adjacent lanes provide geometric passing space.",
    }
    output = args.config.resolve().parent / "scenario_selection_wide_lane.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
