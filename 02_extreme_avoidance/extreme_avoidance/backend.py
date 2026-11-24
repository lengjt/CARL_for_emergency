"""Thin native OCD adapter. No fallback to a different physics model."""
import json
import math
from pathlib import Path
import numpy as np
from .scenario import ROOT

PREFIX = "vehicle_dynamics_double_track."

class OCDBackend:
    def __init__(self, scenario, vehicle_file=None):
        try:
            from ._ocd_native import Model, UPSTREAM_COMMIT
        except ImportError as e:
            raise RuntimeError("OCD native binding unavailable; run bash scripts/bootstrap.sh") from e
        self.scenario = scenario
        self.vehicle_file = Path(vehicle_file or ROOT / "configs/vehicle.json")
        self.config = json.loads(self.vehicle_file.read_text())
        self.params = dict(self.config["ocd_parameters"])
        self.params["integration_step_size_s"] = scenario.dt
        self.commit = UPSTREAM_COMMIT
        self.model_type = Model
        self.mass = self.params[PREFIX+"mass_vehicle_kg"]
        self.lf = self.params[PREFIX+"cog.distance_from_front_axle_m"]
        self.wheelbase = self.params[PREFIX+"wheelbase_m"]
        self.lr = self.wheelbase-self.lf
        self.track_f = self.params[PREFIX+"track_width_m.front"]
        self.track_r = self.params[PREFIX+"track_width_m.rear"]
        self.length = self.config["body"]["length_m"]
        self.width = self.config["body"]["width_m"]
        self.limits = self.config["execution"]
        self.reset()

    def reset(self):
        sc = self.scenario
        x, y, yaw = sc.road.pose(sc.initial_s, sc.initial_d)
        yaw += sc.initial_heading_error
        p = dict(self.params)
        initial = {"x_m": x, "y_m": y, "psi_rad": yaw, "v_x_mps": sc.speed,
                   "v_y_mps": sc.initial_vy, "psi_dot_radps": sc.initial_yaw_rate}
        for key, value in initial.items():
            p["initial_state.vehicle_dynamics."+key] = float(value)
        rf = p[PREFIX+"tire.rolling_radius_m.front"]
        rr = p[PREFIX+"tire.rolling_radius_m.rear"]
        for key, value in {"omega_FL_radps": (sc.speed-sc.initial_yaw_rate*self.track_f/2)/rf,
                           "omega_FR_radps": (sc.speed+sc.initial_yaw_rate*self.track_f/2)/rf,
                           "omega_rear_axle_radps": sc.speed/rr,
                           "omega_diff_rear_radps": -sc.initial_yaw_rate*self.track_r/(2*rr)}.items():
            p["initial_state.drivetrain."+key] = float(value)
        self.initial_parameters = p
        self.native = self.model_type(p)  # fresh object avoids residual inputs/caches
        self.time = 0.0
        self.state = {**initial, "phi_rad": 0.0, "theta_rad": 0.0, "steering_rad": 0.0,
                      "ax_mps2": 0.0, "ay_mps2": 0.0}
        for k in ("fx_N", "fy_N", "fz_N", "kappa", "alpha_rad", "omega_radps", "load_torque_Nm"):
            self.state[k] = [0.0]*4
        self.force = 0.0
        self.executed_force = 0.0
        self.abs_factor = 1.0
        self.command = (0.0, 0.0)
        self.applied_command = (0.0, 0.0)
        self.command_queue = []
        self.last_scale = self.wheel_scales()
        return dict(self.state)

    def wheel_positions(self):
        x, y, yaw = (self.state[k] for k in ("x_m", "y_m", "psi_rad"))
        local = np.array([[self.lf, self.track_f/2], [self.lf, -self.track_f/2],
                          [-self.lr, self.track_r/2], [-self.lr, -self.track_r/2]])
        c, s = math.cos(yaw), math.sin(yaw)
        return local @ np.array([[c, s], [-s, c]]) + [x, y]

    def wheel_scales(self):
        return [self.scenario.scale_at(x, y, self.time) for x, y in self.wheel_positions()]

    def request(self, steer, force):
        if not math.isfinite(steer) or not math.isfinite(force):
            raise ValueError("Nonfinite action")
        angle_max = self.params["steering_actuator.angle_max_rad"]
        self.command = (float(np.clip(steer, -angle_max, angle_max)),
                        float(np.clip(force, -self.limits["max_brake_force_N"], self.limits["max_drive_force_N"])))
        self.command_queue.append((self.time+self.limits["command_delay_s"], self.command))

    def step(self):
        dt = self.scenario.dt
        while self.command_queue and self.command_queue[0][0] <= self.time+1e-10:
            _, self.applied_command = self.command_queue.pop(0)
        steer, desired_force = self.applied_command
        # Common lumped force-request actuator for every controller; OCD maps to wheel torque.
        delta = (desired_force-self.force)*(1-math.exp(-dt/self.limits["force_time_constant_s"]))
        rate = self.limits["force_rate_Nps"]*dt
        self.force += float(np.clip(delta, -rate, rate))
        # Simple shared brake-slip regulator, not a production four-channel ABS.
        if self.limits["slip_regulator"] and self.state["v_x_mps"] > 2 and self.force < 0:
            slip = -min(self.state["kappa"])
            change = -15*dt if slip > 0.18 else (3*dt if slip < 0.10 else 0.0)
            self.abs_factor = float(np.clip(self.abs_factor+change, 0.05, 1.0))
        else:
            self.abs_factor = 1.0
        self.last_scale = self.wheel_scales()
        self.executed_force = self.force*self.abs_factor if self.force < 0 else self.force
        next_state = self.native.step(steer, self.executed_force, self.last_scale,
                                      self.scenario.friction_mode == "internal_mu")
        if not all(np.all(np.isfinite(v)) for v in next_state.values()):
            raise FloatingPointError("OCD produced a nonfinite state")
        self.state = next_state
        self.time += dt
        return self.state
