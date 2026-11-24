#!/usr/bin/env python3
"""nuPlan motion-model adapter for the project's existing Open Car Dynamics backend."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
OCD_PROJECT = RESEARCH / "02_extreme_avoidance"
FLOW_NUPLAN = RESEARCH / "01_flow_planner/upstream/nuplan-devkit"
sys.path[:0] = [str(OCD_PROJECT), str(FLOW_NUPLAN)]

from extreme_avoidance.backend import OCDBackend, PREFIX  # noqa: E402
from extreme_avoidance.scenario import Scenario  # noqa: E402
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState  # noqa: E402
from nuplan.common.actor_state.ego_state import EgoState, EgoStateDot  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint  # noqa: E402
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters  # noqa: E402
from nuplan.common.geometry.compute import principal_value  # noqa: E402
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import AbstractMotionModel  # noqa: E402


class OCDMotionModel(AbstractMotionModel):
    """Execute LQR commands through OCD and return a nuPlan rear-axle EgoState."""

    def __init__(self, vehicle: VehicleParameters, physics_dt: float = 0.001, base_scale: float = 1.0):
        self._vehicle = vehicle
        self.physics_dt = float(physics_dt)
        self.base_scale = float(base_scale)
        self.backend = None
        self.origin_world_cg = None
        self.origin_heading = None
        self.ocd_lr = None
        self.effective_mass = None
        self.log = []

    def reset(self):
        self.backend = None
        self.origin_world_cg = None
        self.origin_heading = None
        self.ocd_lr = None
        self.effective_mass = None
        self.log = []

    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        raise NotImplementedError("OCD is integrated internally; an external state derivative is not exposed")

    def _initialize(self, state: EgoState) -> None:
        vehicle_config = json.loads((OCD_PROJECT / "configs/vehicle.json").read_text())
        params = vehicle_config["ocd_parameters"]
        lf = float(params[PREFIX + "cog.distance_from_front_axle_m"])
        wheelbase = float(params[PREFIX + "wheelbase_m"])
        self.ocd_lr = wheelbase - lf
        wheel_mass_front = float(params[PREFIX + "mass_wheel_kg.front"])
        wheel_mass_rear = float(params[PREFIX + "mass_wheel_kg.rear"])
        self.effective_mass = float(params[PREFIX + "mass_vehicle_kg"]) + 2 * wheel_mass_front + 2 * wheel_mass_rear

        yaw_rate = float(state.dynamic_car_state.angular_velocity)
        rear_velocity = state.dynamic_car_state.rear_axle_velocity_2d
        cg_vy = float(rear_velocity.y + yaw_rate * self.ocd_lr)
        self.origin_heading = float(state.rear_axle.heading)
        self.origin_world_cg = np.asarray(
            [
                state.rear_axle.x + self.ocd_lr * math.cos(self.origin_heading),
                state.rear_axle.y + self.ocd_lr * math.sin(self.origin_heading),
            ],
            dtype=float,
        )
        scenario = Scenario(
            name="nuplan_ocd_adapter",
            speed=max(0.0, float(rear_velocity.x)),
            initial_vy=cg_vy,
            initial_yaw_rate=yaw_rate,
            dt=self.physics_dt,
            control_dt=0.1,
            duration=100.0,
            base_scale=self.base_scale,
            friction_mode="internal_mu",
            obstacles=(),
        )
        self.backend = OCDBackend(scenario, OCD_PROJECT / "configs/vehicle.json")
        initial_steering = float(state.tire_steering_angle)
        if abs(initial_steering) > 1e-8:
            raise ValueError(
                f"Selected scenario has nonzero initial steering {initial_steering}; "
                "the existing OCDBackend reset contract initializes steering at zero"
            )

    def _drag_compensation(self, vx: float) -> float:
        p = self.backend.params
        c_rr = float(p[PREFIX + "tire.rolling_resistance_coefficient"])
        rho = float(p[PREFIX + "aerodynamics.air_density_kgpm3"])
        area = float(p[PREFIX + "aerodynamics.A_m2"])
        cd = float(p[PREFIX + "aerodynamics.c_d"])
        if abs(vx) < 0.05:
            return 0.0
        magnitude = c_rr * self.effective_mass * 9.81 + 0.5 * rho * area * cd * vx * vx
        return math.copysign(magnitude, vx)

    def propagate_state(
        self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint
    ) -> EgoState:
        if self.backend is None:
            self._initialize(state)
        dt_control = float(sampling_time.time_s)
        substeps = int(round(dt_control / self.physics_dt))
        integration_duration = substeps * self.physics_dt
        # nuPlan timestamps are integer microseconds and this log's nominal 100 ms
        # period alternates by a few tens of microseconds. OCD keeps its validated
        # fixed 1 ms step; the nearest integer substep count preserves the horizon.
        if substeps <= 0 or abs(integration_duration - dt_control) > self.physics_dt / 2:
            raise ValueError(
                f"Control period {dt_control} cannot be represented closely by OCD step {self.physics_dt}"
            )

        acceleration_command = float(ideal_dynamic_state.rear_axle_acceleration_2d.x)
        steering_rate_command = float(ideal_dynamic_state.tire_steering_rate)
        measured_steering = float(self.backend.state["steering_rad"])
        steering_request = measured_steering + steering_rate_command * dt_control
        vx_before = float(self.backend.state["v_x_mps"])
        drag_compensation = self._drag_compensation(vx_before)
        raw_force_request = self.effective_mass * acceleration_command + drag_compensation
        self.backend.request(steering_request, raw_force_request)

        previous_yaw_rate = float(self.backend.state["psi_dot_radps"])
        previous_steering = measured_steering
        for _ in range(substeps):
            self.backend.step()
        ocd = self.backend.state

        c0, s0 = math.cos(self.origin_heading), math.sin(self.origin_heading)
        local_x, local_y = float(ocd["x_m"]), float(ocd["y_m"])
        cg_x = self.origin_world_cg[0] + c0 * local_x - s0 * local_y
        cg_y = self.origin_world_cg[1] + s0 * local_x + c0 * local_y
        world_heading = principal_value(self.origin_heading + float(ocd["psi_rad"]))
        rear_x = cg_x - self.ocd_lr * math.cos(world_heading)
        rear_y = cg_y - self.ocd_lr * math.sin(world_heading)

        yaw_rate = float(ocd["psi_dot_radps"])
        yaw_acceleration = (yaw_rate - previous_yaw_rate) / dt_control
        steering = float(ocd["steering_rad"])
        steering_rate = (steering - previous_steering) / dt_control
        rear_vx = float(ocd["v_x_mps"])
        rear_vy = float(ocd["v_y_mps"] - yaw_rate * self.ocd_lr)
        rear_ax = float(ocd["ax_mps2"] + yaw_rate * yaw_rate * self.ocd_lr)
        rear_ay = float(ocd["ay_mps2"] - yaw_acceleration * self.ocd_lr)

        self.log.append(
            {
                "timestamp_us": int((state.time_point + sampling_time).time_us),
                "control_period_s": dt_control,
                "physics_substeps": substeps,
                "physics_integrated_duration_s": integration_duration,
                "time_quantization_error_s": integration_duration - dt_control,
                "lqr_acceleration_command_mps2": acceleration_command,
                "lqr_steering_rate_command_radps": steering_rate_command,
                "steering_request_rad": float(self.backend.command[0]),
                "steering_applied_request_rad": float(self.backend.applied_command[0]),
                "steering_actual_rad": steering,
                "force_request_raw_N": raw_force_request,
                "force_request_clipped_N": float(self.backend.command[1]),
                "force_applied_request_N": float(self.backend.applied_command[1]),
                "force_executed_N": float(self.backend.executed_force),
                "drag_compensation_N": drag_compensation,
                "ocd_cg_x_local_m": local_x,
                "ocd_cg_y_local_m": local_y,
                "ocd_heading_local_rad": float(ocd["psi_rad"]),
                "ocd_vx_mps": float(ocd["v_x_mps"]),
                "ocd_vy_mps": float(ocd["v_y_mps"]),
                "ocd_yaw_rate_radps": yaw_rate,
                "ocd_ax_mps2": float(ocd["ax_mps2"]),
                "ocd_ay_mps2": float(ocd["ay_mps2"]),
                "ocd_peak_scale_fl": float(self.backend.last_scale[0]),
                "ocd_peak_scale_fr": float(self.backend.last_scale[1]),
                "ocd_peak_scale_rl": float(self.backend.last_scale[2]),
                "ocd_peak_scale_rr": float(self.backend.last_scale[3]),
            }
        )
        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(rear_x, rear_y, world_heading),
            rear_axle_velocity_2d=StateVector2D(rear_vx, rear_vy),
            rear_axle_acceleration_2d=StateVector2D(rear_ax, rear_ay),
            tire_steering_angle=steering,
            time_point=state.time_point + sampling_time,
            vehicle_parameters=self._vehicle,
            is_in_auto_mode=True,
            angular_vel=yaw_rate,
            angular_accel=yaw_acceleration,
            tire_steering_rate=steering_rate,
        )
