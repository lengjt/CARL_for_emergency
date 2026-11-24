#!/usr/bin/env python3
"""Short direction and magnitude checks for the nuPlan-to-OCD adapter."""
from __future__ import annotations

import json
from pathlib import Path

from ocd_motion_model import OCDMotionModel
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

HERE = Path(__file__).resolve().parent


def initial_state():
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(0.0, 0.0, 0.0),
        rear_axle_velocity_2d=StateVector2D(5.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(0),
        vehicle_parameters=get_pacifica_parameters(),
    )


def run(acceleration, steering_rate, steps=10):
    vehicle = get_pacifica_parameters()
    model = OCDMotionModel(vehicle)
    state = initial_state()
    for _ in range(steps):
        command = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(acceleration, 0.0),
            tire_steering_rate=steering_rate,
        )
        state = model.propagate_state(state, command, TimePoint(100_000))
    return {
        "x_m": float(state.rear_axle.x),
        "y_m": float(state.rear_axle.y),
        "heading_rad": float(state.rear_axle.heading),
        "speed_mps": float(state.dynamic_car_state.speed),
        "steering_rad": float(state.tire_steering_angle),
        "yaw_rate_radps": float(state.dynamic_car_state.angular_velocity),
    }


def main():
    result = {
        "straight_hold": run(0.0, 0.0),
        "positive_left_turn": run(0.0, 0.10),
        "braking": run(-2.0, 0.0),
    }
    checks = {
        "straight_heading_small": abs(result["straight_hold"]["heading_rad"]) < 1e-3,
        "straight_lateral_small": abs(result["straight_hold"]["y_m"]) < 1e-2,
        "positive_steering_turns_left": result["positive_left_turn"]["y_m"] > 0.0
        and result["positive_left_turn"]["heading_rad"] > 0.0,
        "braking_reduces_speed": result["braking"]["speed_mps"] < result["straight_hold"]["speed_mps"] - 1.0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    output = {"passed": all(checks.values()), "checks": checks, "results_after_1s": result}
    (HERE / "interface_check.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
