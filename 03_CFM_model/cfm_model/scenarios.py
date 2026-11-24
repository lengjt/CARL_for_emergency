"""Deterministic scenario expansion and leakage-resistant family splitting."""

import hashlib
import itertools
from dataclasses import asdict


VARIANTS = (
    {"road_width": 10.0, "obstacle_s_delta": 0.0, "obstacle_d": 0.0,
     "obstacle_length": 4.0, "obstacle_width": 1.8, "initial_d": 0.0,
     "initial_heading": 0.0, "initial_vy": 0.0, "initial_yaw_rate": 0.0},
    {"road_width": 9.6, "obstacle_s_delta": 0.0, "obstacle_d": 0.15,
     "obstacle_length": 4.2, "obstacle_width": 1.9, "initial_d": 0.05,
     "initial_heading": 0.004, "initial_vy": 0.0, "initial_yaw_rate": 0.0},
    {"road_width": 10.4, "obstacle_s_delta": 0.0, "obstacle_d": -0.15,
     "obstacle_length": 3.8, "obstacle_width": 1.7, "initial_d": -0.05,
     "initial_heading": -0.004, "initial_vy": 0.0, "initial_yaw_rate": 0.0},
    {"road_width": 9.8, "obstacle_s_delta": 1.5, "obstacle_d": 0.1,
     "obstacle_length": 4.0, "obstacle_width": 1.8, "initial_d": 0.0,
     "initial_heading": 0.0, "initial_vy": 0.10, "initial_yaw_rate": 0.01},
    {"road_width": 10.2, "obstacle_s_delta": -1.5, "obstacle_d": -0.1,
     "obstacle_length": 4.0, "obstacle_width": 1.8, "initial_d": 0.0,
     "initial_heading": 0.0, "initial_vy": -0.10, "initial_yaw_rate": -0.01},
)


def family_id(speed, scale, reveal):
    return f"v{speed:g}_g{scale:g}_d{reveal:g}"


def split_families(config):
    families = [family_id(*values) for values in itertools.product(
        config["speeds_mps"], config["capability_scales"], config["reveal_distances_m"])]
    seed = str(config["split_seed"])
    ordered = sorted(families, key=lambda x: hashlib.sha256((seed + ":" + x).encode()).hexdigest())
    counts = config["split_family_counts"]
    if sum(counts.values()) != len(ordered):
        raise ValueError("split_family_counts must cover every family")
    result = {}
    start = 0
    for split in ("train", "val", "test"):
        for name in ordered[start:start + counts[split]]:
            result[name] = split
        start += counts[split]
    return result


def build_scenarios(config, scenario_types):
    Scenario, Road, Obstacle = scenario_types
    assignments = split_families(config)
    rows = []
    for speed, scale, reveal in itertools.product(
            config["speeds_mps"], config["capability_scales"], config["reveal_distances_m"]):
        family = family_id(speed, scale, reveal)
        for index, variant in enumerate(VARIANTS[:config["variants_per_family"]]):
            obstacle_s = config["obstacle_s_m"] + variant["obstacle_s_delta"]
            scenario = Scenario(
                name=f"{family}_p{index}", seed=config["split_seed"] + index,
                speed=speed, initial_d=variant["initial_d"],
                initial_heading_error=variant["initial_heading"],
                initial_vy=variant["initial_vy"], initial_yaw_rate=variant["initial_yaw_rate"],
                dt=config["physics_dt_s"], control_dt=config["control_dt_s"],
                duration=config["duration_s"], base_scale=scale,
                friction_mode=config["friction_mode"],
                road=Road(width=variant["road_width"], length=config["road_length_m"]),
                obstacles=(Obstacle(s=obstacle_s, d=variant["obstacle_d"],
                                    length=variant["obstacle_length"], width=variant["obstacle_width"],
                                    reveal_distance=reveal),))
            rows.append({"group_id": scenario.name, "family_id": family,
                         "variant_id": index, "split": assignments[family],
                         "scenario": asdict(scenario)})
    return rows
