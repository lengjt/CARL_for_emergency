#!/usr/bin/env python3
"""Constrained optimal-control expert for the interPlan/OCD comparison.

The optimizer deliberately uses a model that is simpler than OCD.  It produces
time-parameterized rear-axle states; those states are later converted to a
nuPlan trajectory and tracked by the unchanged LQR/OCD chain.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import casadi as ca
import numpy as np
from shapely.geometry import LineString
from shapely.ops import transform, unary_union


@dataclass(frozen=True)
class ExpertConfig:
    maneuver: str = "safe_stop"
    horizon_s: float = 9.0
    integration_dt_s: float = 0.2
    output_dt_s: float = 0.1
    wheelbase_m: float = 3.2
    rear_axle_to_center_m: float = 1.461
    vehicle_length_m: float = 5.176
    vehicle_width_m: float = 2.297
    road_margin_m: float = 0.30
    obstacle_margin_m: float = 0.05
    obstacle_longitudinal_tracking_buffer_m: float = 12.0
    pass_longitudinal_tracking_buffer_m: float = 5.0
    tracking_buffer_reference_speed_mps: float = 15.0
    obstacle_zone_speed_limit_mps: float = 7.0
    obstacle_zone_half_length_m: float = 20.0
    maximum_steering_rad: float = 0.30
    maximum_steering_rate_radps: float = 0.50
    minimum_acceleration_mps2: float = -9.5
    maximum_acceleration_mps2: float = 2.5
    maximum_speed_mps: float = 22.0
    capability_scale: float = 1.0
    effective_acceleration_at_scale_1_mps2: float = 9.81
    desired_speed_mps: float = 12.0
    ipopt_max_iterations: int = 500
    ipopt_tolerance: float = 1e-5

    @property
    def intervals(self) -> int:
        count = self.horizon_s / self.integration_dt_s
        if abs(count - round(count)) > 1e-9:
            raise ValueError("horizon must be an integer multiple of integration dt")
        return int(round(count))


@dataclass
class ExpertSolution:
    success: bool
    status: str
    solve_time_s: float
    iterations: int
    objective: float | None
    max_constraint_violation: float | None
    states: np.ndarray | None
    controls: np.ndarray | None
    sampled_states: np.ndarray | None
    message: str = ""


def _principal(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


class SceneGeometry:
    """Frozen local frame, route reference, road corridor and stopped vehicle."""

    def __init__(self, scenario: Any, obstacle_pose: Any):
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        initial = scenario.initial_ego_state.rear_axle
        self.origin = np.asarray([initial.x, initial.y], dtype=float)
        self.heading = float(initial.heading)
        self._c = math.cos(self.heading)
        self._s = math.sin(self.heading)

        route_polygons = []
        for object_id in scenario.get_route_roadblock_ids():
            roadblock = scenario.map_api.get_map_object(object_id, SemanticMapLayer.ROADBLOCK)
            if roadblock is None:
                roadblock = scenario.map_api.get_map_object(object_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if roadblock is not None:
                route_polygons.append(roadblock.polygon)
        if not route_polygons:
            raise RuntimeError("No route roadblock polygons were found")

        def to_local(x, y, z=None):
            dx, dy = np.asarray(x) - self.origin[0], np.asarray(y) - self.origin[1]
            return self._c * dx + self._s * dy, -self._s * dx + self._c * dy

        self.route_polygon_local = transform(to_local, unary_union(route_polygons))
        self.x_grid = np.arange(-20.0, 281.0, 1.0)

        route_states = [scenario.initial_ego_state] + list(scenario.get_expert_ego_trajectory())
        route_world = np.asarray([[state.rear_axle.x, state.rear_axle.y] for state in route_states])
        route_local = self.world_to_local(route_world)
        order = np.argsort(route_local[:, 0])
        route_x = route_local[order, 0]
        route_y = route_local[order, 1]
        route_x, unique = np.unique(route_x, return_index=True)
        route_y = route_y[unique]
        self.reference_y = np.interp(self.x_grid, route_x, route_y)

        lower, upper = [], []
        previous = 0.0
        for x, reference_y in zip(self.x_grid, self.reference_y):
            cross_section = self.route_polygon_local.intersection(LineString([(x, -40.0), (x, 40.0)]))
            segments = []
            geometries = [cross_section] if cross_section.geom_type == "LineString" else getattr(cross_section, "geoms", [])
            for geometry in geometries:
                if geometry.geom_type == "LineString" and geometry.length > 0.05:
                    segments.append((geometry.bounds[1], geometry.bounds[3]))
            if not segments:
                lower.append(np.nan)
                upper.append(np.nan)
                continue
            containing = [item for item in segments if item[0] - 0.2 <= reference_y <= item[1] + 0.2]
            candidates = containing or segments
            selected = min(candidates, key=lambda item: abs(0.5 * (item[0] + item[1]) - previous))
            lower.append(selected[0])
            upper.append(selected[1])
            previous = 0.5 * (selected[0] + selected[1])
        lower, upper = np.asarray(lower), np.asarray(upper)
        valid = np.isfinite(lower) & np.isfinite(upper)
        if valid.sum() < 20:
            raise RuntimeError("Route polygon did not produce a usable road corridor")
        self.lower = np.interp(self.x_grid, self.x_grid[valid], lower[valid])
        self.upper = np.interp(self.x_grid, self.x_grid[valid], upper[valid])

        obstacle_world = np.asarray([[obstacle_pose.x, obstacle_pose.y]], dtype=float)
        obstacle_local = self.world_to_local(obstacle_world)[0]
        self.obstacle_x = float(obstacle_local[0])
        self.obstacle_y = float(obstacle_local[1])
        self.obstacle_heading = float(_principal(obstacle_pose.heading - self.heading))

        tracked = scenario.get_tracked_objects_at_iteration(0).tracked_objects.tracked_objects
        candidates = [
            item for item in tracked
            if math.hypot(item.center.x - obstacle_pose.x, item.center.y - obstacle_pose.y) < 0.5
        ]
        if not candidates:
            raise RuntimeError("Unable to recover stopped-vehicle dimensions")
        obstacle = min(candidates, key=lambda item: math.hypot(item.center.x - obstacle_pose.x, item.center.y - obstacle_pose.y))
        self.obstacle_length = float(obstacle.box.length)
        self.obstacle_width = float(obstacle.box.width)

    def world_to_local(self, points: np.ndarray) -> np.ndarray:
        delta = np.asarray(points, dtype=float) - self.origin
        return np.column_stack((self._c * delta[:, 0] + self._s * delta[:, 1],
                                -self._s * delta[:, 0] + self._c * delta[:, 1]))

    def local_to_world(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return self.origin + np.column_stack((self._c * points[:, 0] - self._s * points[:, 1],
                                              self._s * points[:, 0] + self._c * points[:, 1]))

    def initial_vector(self, ego_state: Any) -> np.ndarray:
        local = self.world_to_local(np.asarray([[ego_state.rear_axle.x, ego_state.rear_axle.y]]))[0]
        return np.asarray([
            local[0], local[1], float(_principal(ego_state.rear_axle.heading - self.heading)),
            max(0.0, ego_state.dynamic_car_state.speed), ego_state.tire_steering_angle,
        ])


class OptimalControlExpert:
    """Direct multiple-shooting optimal-control problem with a fixed pass side."""

    def __init__(self, geometry: SceneGeometry, config: ExpertConfig, pass_side: int = -1):
        if pass_side not in (-1, 1):
            raise ValueError("pass_side must be -1 (right) or 1 (left)")
        self.geometry = geometry
        self.config = config
        self.pass_side = pass_side
        self.previous_states: np.ndarray | None = None
        self.previous_controls: np.ndarray | None = None
        self._build_problem()

    def _build_problem(self) -> None:
        cfg, geo = self.config, self.geometry
        n, dt = cfg.intervals, cfg.integration_dt_s
        opti = ca.Opti()
        state = opti.variable(5, n + 1)  # x, y, heading, speed, steering
        control = opti.variable(2, n)  # longitudinal acceleration, steering rate
        initial = opti.parameter(5)
        stop_line_parameter = opti.parameter() if cfg.maneuver == "safe_stop" else None
        opti.subject_to(state[:, 0] == initial)

        road_lower = ca.interpolant("road_lower", "linear", [geo.x_grid], geo.lower)
        road_upper = ca.interpolant("road_upper", "linear", [geo.x_grid], geo.upper)
        route_y = ca.interpolant("route_y", "linear", [geo.x_grid], geo.reference_y)

        def dynamics(value, action):
            x, y, heading, speed, steering = ca.vertsplit(value)
            acceleration, steering_rate = ca.vertsplit(action)
            return ca.vertcat(
                speed * ca.cos(heading), speed * ca.sin(heading),
                speed * ca.tan(steering) / cfg.wheelbase_m,
                acceleration, steering_rate,
            )

        def rk4(value, action):
            k1 = dynamics(value, action)
            k2 = dynamics(value + 0.5 * dt * k1, action)
            k3 = dynamics(value + 0.5 * dt * k2, action)
            k4 = dynamics(value + dt * k3, action)
            return value + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        opti.subject_to(opti.bounded(0.0, state[3, :], cfg.maximum_speed_mps))
        opti.subject_to(opti.bounded(-cfg.maximum_steering_rad, state[4, :], cfg.maximum_steering_rad))
        opti.subject_to(opti.bounded(cfg.minimum_acceleration_mps2, control[0, :], cfg.maximum_acceleration_mps2))
        opti.subject_to(opti.bounded(-cfg.maximum_steering_rate_radps, control[1, :], cfg.maximum_steering_rate_radps))

        half_width = 0.5 * cfg.vehicle_width_m
        half_length = 0.5 * cfg.vehicle_length_m
        obstacle_a_physical = 0.5 * (cfg.vehicle_length_m + geo.obstacle_length) + cfg.obstacle_margin_m
        obstacle_a = obstacle_a_physical + cfg.pass_longitudinal_tracking_buffer_m
        obstacle_b = 0.5 * (cfg.vehicle_width_m + geo.obstacle_width) + cfg.obstacle_margin_m
        # The expansion makes the eighth-order superellipse enclose the axis-aligned
        # combined rectangle rather than cutting through its corners.
        expansion = 2.0 ** (1.0 / 8.0)
        obstacle_a_physical *= expansion
        obstacle_a *= expansion
        obstacle_b *= expansion
        co, so = math.cos(geo.obstacle_heading), math.sin(geo.obstacle_heading)
        maximum_combined_acceleration = cfg.effective_acceleration_at_scale_1_mps2 * cfg.capability_scale

        objective = 0
        constraints_for_audit = []
        for index in range(n):
            opti.subject_to(state[:, index + 1] == rk4(state[:, index], control[:, index]))
            # Enforce geometry on the next predicted state.  The measured initial
            # state cannot be changed and may already lie inside a conservative
            # planning buffer even though it is not in physical collision.
            path_state = state[:, index + 1]
            center_x = path_state[0] + cfg.rear_axle_to_center_m * ca.cos(path_state[2])
            center_y = path_state[1] + cfg.rear_axle_to_center_m * ca.sin(path_state[2])
            lateral_footprint_extent = (
                half_width * ca.sqrt(ca.cos(path_state[2]) ** 2 + 1e-8)
                + half_length * ca.sqrt(ca.sin(path_state[2]) ** 2 + 1e-8)
                + cfg.road_margin_m
            )
            road_low_margin = center_y - (road_lower(center_x) + lateral_footprint_extent)
            road_high_margin = (road_upper(center_x) - lateral_footprint_extent) - center_y
            opti.subject_to(road_low_margin >= 0)
            opti.subject_to(road_high_margin >= 0)
            constraints_for_audit.extend([road_low_margin, road_high_margin])

            dx, dy = center_x - geo.obstacle_x, center_y - geo.obstacle_y
            obstacle_dx = co * dx + so * dy
            obstacle_dy = -so * dx + co * dy
            if cfg.maneuver == "safe_stop":
                # Reserve longitudinal distance for the unchanged LQR, OCD command
                # delay and force dynamics; model-internal stopping at the physical
                # contact line was not closed-loop trackable in the pilot run.
                stop_margin = stop_line_parameter - center_x
                opti.subject_to(stop_margin >= 0)
                constraints_for_audit.append(stop_margin)
            elif cfg.maneuver == "pass":
                separation = (obstacle_dx / obstacle_a) ** 8 + (obstacle_dy / obstacle_b) ** 8 - 1.0
                opti.subject_to(separation >= 0)
                constraints_for_audit.append(separation)
            else:
                raise ValueError(f"Unknown expert maneuver: {cfg.maneuver}")

            if cfg.maneuver == "pass":
                speed_gate = ca.exp(-((center_x - geo.obstacle_x) / cfg.obstacle_zone_half_length_m) ** 8)
                local_speed_limit = (cfg.obstacle_zone_speed_limit_mps
                                     + (cfg.maximum_speed_mps - cfg.obstacle_zone_speed_limit_mps) * (1.0 - speed_gate))
                speed_margin = local_speed_limit - path_state[3]
                opti.subject_to(speed_margin >= 0)
                constraints_for_audit.append(speed_margin)

            lateral_acceleration = state[3, index] ** 2 * ca.tan(state[4, index]) / cfg.wheelbase_m
            combined_margin = maximum_combined_acceleration ** 2 - (
                control[0, index] ** 2 + lateral_acceleration ** 2
            )
            opti.subject_to(combined_margin >= 0)
            constraints_for_audit.append(combined_margin)

            center_error = state[1, index] - route_y(state[0, index])
            speed_cost = ((state[3, index] - cfg.desired_speed_mps) ** 2
                          if cfg.maneuver == "pass" else 0.02 * state[3, index] ** 2)
            objective += dt * (
                0.18 * center_error ** 2
                + 0.18 * state[2, index] ** 2
                + 0.06 * speed_cost
                + 0.20 * state[4, index] ** 2
                + 0.045 * control[0, index] ** 2
                + 0.30 * control[1, index] ** 2
            )
            if index:
                objective += 0.04 * (control[0, index] - control[0, index - 1]) ** 2
                objective += 0.08 * (control[1, index] - control[1, index - 1]) ** 2

        final_center_error = state[1, -1] - route_y(state[0, -1])
        objective += 4.0 * final_center_error ** 2 + 2.0 * state[2, -1] ** 2
        if cfg.maneuver == "safe_stop":
            opti.subject_to(state[3, -1] <= 0.05)
            opti.subject_to(opti.bounded(-0.02, state[2, -1], 0.02))
            opti.subject_to(opti.bounded(-0.01, state[4, -1], 0.01))
            opti.subject_to(opti.bounded(-0.12, final_center_error, 0.12))
            objective += 20.0 * state[3, -1] ** 2
            objective -= 1.2 * state[0, -1]
        else:
            objective += 0.4 * (state[3, -1] - cfg.desired_speed_mps) ** 2
            objective -= 0.35 * state[0, -1]
        opti.minimize(objective)

        options = {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": cfg.ipopt_max_iterations,
                "tol": cfg.ipopt_tolerance,
                "acceptable_tol": 5e-4,
                "acceptable_iter": 8,
                "warm_start_init_point": "yes",
            },
        }
        opti.solver("ipopt", options)
        self.opti, self.state, self.control, self.initial = opti, state, control, initial
        self.stop_line_parameter = stop_line_parameter
        self.objective = objective
        self.audit_constraints = ca.vertcat(*constraints_for_audit)
        self.route_y_values = geo.reference_y

    def _initial_guess(self, initial: np.ndarray, warm_shift_steps: int = 1) -> tuple[np.ndarray, np.ndarray]:
        cfg, geo = self.config, self.geometry
        n, dt = cfg.intervals, cfg.integration_dt_s
        if self.previous_states is not None and self.previous_controls is not None:
            shift = min(max(1, int(warm_shift_steps)), n)
            states = np.column_stack((self.previous_states[:, shift:], np.repeat(
                self.previous_states[:, -1:], shift, axis=1)))
            controls = np.column_stack((self.previous_controls[:, shift:], np.repeat(
                self.previous_controls[:, -1:], shift, axis=1)))
            states[:, 0] = initial
            return states, controls

        times = np.arange(n + 1) * dt
        if cfg.maneuver == "safe_stop":
            center0 = initial[0] + cfg.rear_axle_to_center_m * math.cos(initial[2])
            stop_line = self._stop_line(initial[3])
            distance = max(2.0, stop_line - center0)
            deceleration = min(abs(cfg.minimum_acceleration_mps2),
                               initial[3] ** 2 / (2.0 * distance) * 1.15)
            speeds = np.maximum(0.0, initial[3] - deceleration * times)
        else:
            target_speed = min(initial[3], cfg.obstacle_zone_speed_limit_mps)
            speeds = target_speed + (initial[3] - target_speed) * np.exp(-times / 1.2)
        x = initial[0] + np.cumsum(np.r_[0.0, 0.5 * (speeds[:-1] + speeds[1:]) * dt])
        center = np.interp(x, geo.x_grid, geo.reference_y)
        road_low = np.interp(x, geo.x_grid, geo.lower) + 0.5 * cfg.vehicle_width_m + cfg.road_margin_m + 0.03
        road_high = np.interp(x, geo.x_grid, geo.upper) - 0.5 * cfg.vehicle_width_m - cfg.road_margin_m - 0.03
        if cfg.maneuver == "safe_stop":
            y = np.clip(center, road_low, road_high)
        else:
            pass_target = geo.obstacle_y + self.pass_side * (
                0.5 * (cfg.vehicle_width_m + geo.obstacle_width) + cfg.obstacle_margin_m + 0.25
            )
            route_at_obstacle = float(np.interp(geo.obstacle_x, geo.x_grid, geo.reference_y))
            amplitude = pass_target - route_at_obstacle
            bump = np.exp(-((x - geo.obstacle_x) / 15.0) ** 4)
            y = np.clip(center + amplitude * bump, road_low, road_high)
        y[0] = initial[1]
        heading = np.arctan2(np.gradient(y), np.maximum(np.gradient(x), 1e-3))
        heading[0] = initial[2]
        curvature = np.gradient(np.unwrap(heading)) / np.maximum(np.gradient(x), 1e-3)
        steering = np.clip(np.arctan(cfg.wheelbase_m * curvature),
                           -cfg.maximum_steering_rad, cfg.maximum_steering_rad)
        steering[0] = initial[4]
        states = np.vstack((x, y, heading, speeds, steering))
        acceleration = np.clip(np.diff(speeds) / dt, cfg.minimum_acceleration_mps2, cfg.maximum_acceleration_mps2)
        steering_rate = np.clip(np.diff(steering) / dt,
                                -cfg.maximum_steering_rate_radps, cfg.maximum_steering_rate_radps)
        return states, np.vstack((acceleration, steering_rate))

    def solve(self, initial: np.ndarray, warm_shift_s: float | None = None) -> ExpertSolution:
        initial = np.asarray(initial, dtype=float)
        warm_shift_steps = 1 if warm_shift_s is None else int(round(warm_shift_s / self.config.integration_dt_s))
        states_guess, controls_guess = self._initial_guess(initial, warm_shift_steps)
        self.opti.set_value(self.initial, initial)
        if self.stop_line_parameter is not None:
            self.opti.set_value(self.stop_line_parameter, self._stop_line(initial[3]))
        self.opti.set_initial(self.state, states_guess)
        self.opti.set_initial(self.control, controls_guess)
        started = time.perf_counter()
        try:
            solution = self.opti.solve()
            elapsed = time.perf_counter() - started
            states = np.asarray(solution.value(self.state), dtype=float)
            controls = np.asarray(solution.value(self.control), dtype=float)
            stats = self.opti.stats()
            self.previous_states, self.previous_controls = states, controls
            sampled = self._resample(states, controls)
            return ExpertSolution(
                success=True,
                status=str(stats.get("return_status", "Solve_Succeeded")),
                solve_time_s=elapsed,
                iterations=int(stats.get("iter_count", 0)),
                objective=float(solution.value(self.objective)),
                max_constraint_violation=0.0,
                states=states,
                controls=controls,
                sampled_states=sampled,
            )

        except RuntimeError as error:
            elapsed = time.perf_counter() - started
            stats = self.opti.stats()
            return ExpertSolution(
                success=False,
                status=str(stats.get("return_status", "solver_exception")),
                solve_time_s=elapsed,
                iterations=int(stats.get("iter_count", 0)),
                objective=None,
                max_constraint_violation=None,
                states=None,
                controls=None,
                sampled_states=None,
                message=str(error),
            )

    def _stop_line(self, speed_mps: float) -> float:
        cfg, geo = self.config, self.geometry
        expansion = 2.0 ** (1.0 / 8.0)
        physical_clearance = expansion * (
            0.5 * (cfg.vehicle_length_m + geo.obstacle_length) + cfg.obstacle_margin_m
        )
        return geo.obstacle_x - physical_clearance - cfg.obstacle_longitudinal_tracking_buffer_m

    def _resample(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        cfg = self.config
        source_t = np.arange(cfg.intervals + 1) * cfg.integration_dt_s
        target_t = np.arange(int(round(cfg.horizon_s / cfg.output_dt_s)) + 1) * cfg.output_dt_s
        sampled = np.empty((len(target_t), 7), dtype=float)
        sampled[:, 0] = target_t
        for column in (0, 1, 3, 4):
            sampled[:, column + 1] = np.interp(target_t, source_t, states[column])
        sampled[:, 3] = np.interp(target_t, source_t, np.unwrap(states[2]))
        control_t = source_t[:-1]
        sampled[:, 6] = np.interp(target_t, control_t, controls[0], right=controls[0, -1])
        return sampled

    def metadata(self) -> dict[str, Any]:
        cfg, geo = self.config, self.geometry
        return {
            "method": "CasADi/Ipopt direct multiple shooting",
            "maneuver": cfg.maneuver,
            "prediction_model": "kinematic bicycle with steering state and combined acceleration envelope",
            "state": ["local_rear_axle_x_m", "local_rear_axle_y_m", "heading_rad", "speed_mps", "steering_rad"],
            "control": ["longitudinal_acceleration_mps2", "steering_rate_radps"],
            "horizon_s": cfg.horizon_s,
            "integration_dt_s": cfg.integration_dt_s,
            "output_dt_s": cfg.output_dt_s,
            "capability_scale": cfg.capability_scale,
            "road_margin_m": cfg.road_margin_m,
            "obstacle_margin_m": cfg.obstacle_margin_m,
            "obstacle_longitudinal_tracking_buffer_m": cfg.obstacle_longitudinal_tracking_buffer_m,
            "pass_longitudinal_tracking_buffer_m": cfg.pass_longitudinal_tracking_buffer_m,
            "tracking_buffer_rule": "fixed closed-loop stopping buffer; replanning is suppressed after committing to the stop point",
            "obstacle_zone_speed_limit_mps": cfg.obstacle_zone_speed_limit_mps,
            "pass_side": "right" if self.pass_side == -1 else "left",
            "obstacle_local_pose": [geo.obstacle_x, geo.obstacle_y, geo.obstacle_heading],
            "obstacle_dimensions_m": [geo.obstacle_length, geo.obstacle_width],
            "execution_validation": "unchanged nuPlan LQR followed by OCD double-track MF5.2",
            "model_mismatch": "optimizer feasibility is not treated as OCD closed-loop feasibility",
        }
