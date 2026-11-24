"""Frozen data schema shared by data generation, training, and online inference."""

CONDITION_FIELDS = (
    "relative_obstacle_s_m",
    "relative_obstacle_d_m",
    "road_width_m",
    "capability_scale",
    "v_x_mps",
    "v_y_mps",
    "yaw_rate_radps",
    "steering_rad",
    "actuator_force_kN",
    "lateral_d_m",
    "heading_error_rad",
    "initial_speed_mps",
)

TARGET_FIELDS = ("signed_offset_m", "ramp_length_m", "brake_kN")

TARGET_CLIP = ((-4.0, 4.0), (5.0, 60.0), (0.0, 5.0))


def context_from_row(row, scenario, obstacle):
    """Build causal features available to the planner at one replanning instant."""
    return {
        "relative_obstacle_s_m": float(obstacle.s - row["s"]),
        "relative_obstacle_d_m": float(obstacle.d - row["d"]),
        "road_width_m": float(scenario.road.width),
        "capability_scale": float(min(row[k] for k in (
            "scale_true_fl", "scale_true_fr", "scale_true_rl", "scale_true_rr"))),
        "v_x_mps": float(row["v_x_mps"]),
        "v_y_mps": float(row["v_y_mps"]),
        "yaw_rate_radps": float(row["psi_dot_radps"]),
        "steering_rad": float(row["steering_rad"]),
        "actuator_force_kN": float(row["force_executed_N"] / 1000.0),
        "lateral_d_m": float(row["d"]),
        "heading_error_rad": float(row["heading_error"]),
        "initial_speed_mps": float(scenario.speed),
    }


def target_from_parameters(parameters):
    return {
        "signed_offset_m": float(parameters[0]),
        "ramp_length_m": float(parameters[1]),
        "brake_kN": float(parameters[2] / 1000.0),
    }
