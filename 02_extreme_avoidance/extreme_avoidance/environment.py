"""Deterministic reset/step interface with observation masking and footprint checks."""
import math
import numpy as np
from .backend import OCDBackend
from .scenario import rectangle, separation, road_margin, wrap

class AvoidanceEnv:
    def __init__(self, scenario, vehicle_file=None):
        self.scenario = scenario
        self.backend = OCDBackend(scenario, vehicle_file)
        self.obstacle_polygons = [rectangle(*scenario.road.pose(o.s, o.d), o.length, o.width)
                                  for o in scenario.obstacles]
        self.reset()

    def reset(self):
        self.backend.reset()
        self.revealed = set()
        self.recovery_since = None
        self.passed_obstacles_at = None
        self.safe_stop_at = None
        self.status = "running"
        self.collision = False
        self.offroad = False
        self.min_margin = math.inf
        self.min_gap = math.inf
        self.max_beta = 0.0
        self.max_yaw_rate = 0.0
        self.max_steer = 0.0
        self.max_steer_rate = 0.0
        self.max_force_rate = 0.0
        self.min_actuator_force = 0.0
        self.max_actuator_force = 0.0
        self.scale_history = [(0.0,list(self.backend.last_scale))]
        self.rows = []
        self._reveal()
        self._check_geometry(self.backend.state, self.backend.state)
        if self.collision or self.offroad:
            raise ValueError("Initial vehicle footprint collides or leaves the road")
        self._record()
        return self.observe()

    def _reveal(self):
        s, _, _ = self.scenario.road.project(self.backend.state["x_m"], self.backend.state["y_m"])
        for i, ob in enumerate(self.scenario.obstacles):
            if ob.s-s <= ob.reveal_distance:
                self.revealed.add(i)

    def observe(self):
        b = self.backend
        state = dict(b.state)
        s, d, heading = self.scenario.road.project(state["x_m"], state["y_m"])
        scales = self.scale_history[-1][1] if self.scale_history else b.wheel_scales()
        target_time = b.time-self.scenario.scale_estimate_delay
        for t, old in reversed(self.scale_history):
            if t <= target_time+1e-10:
                scales = old
                break
        else:
            if self.scale_history:
                scales = self.scale_history[0][1]
        estimate = [x*self.scenario.scale_estimate_factor for x in scales]
        return {"time": b.time, "state": state, "s": s, "d": d,
                "road_heading": heading,
                "friction_mode": self.scenario.friction_mode,
                "actuator_force_N": b.force,
                "applied_command": b.applied_command,
                "pending_commands": list(b.command_queue),
                "capability_scale_estimate": estimate,
                "force_scale_estimate": estimate,
                "scale_estimate": estimate,
                "obstacles": [self.scenario.obstacles[i] for i in sorted(self.revealed)]}

    def _check_geometry(self, prev, state):
        b = self.backend
        x, y, yaw = (state[k] for k in ("x_m", "y_m", "psi_rad"))
        # Conservative swept envelope under linear pose interpolation within one physics step.
        dx, dy = x-prev["x_m"], y-prev["y_m"]
        dyaw = wrap(yaw-prev["psi_rad"])
        inflation = 0.5*math.hypot(dx, dy)+0.25*math.hypot(b.length,b.width)*abs(dyaw)
        swept = rectangle((x+prev["x_m"])/2, (y+prev["y_m"])/2,
                          prev["psi_rad"]+dyaw/2, b.length+2*inflation, b.width+2*inflation)
        margin = road_margin(self.scenario.road, swept)
        self.min_margin = min(self.min_margin, margin)
        self.offroad |= margin < 0
        for polygon in self.obstacle_polygons:
            # All physical obstacles are checked, including unrevealed ones.
            gap = separation(swept, polygon)
            self.min_gap = min(self.min_gap, gap)
            self.collision |= gap <= 0

    def _check_recovery(self):
        b, sc = self.backend, self.scenario
        st = b.state
        s, d, heading = sc.road.project(st["x_m"], st["y_m"])
        beta = math.atan2(st["v_y_mps"], max(abs(st["v_x_mps"]), 0.2))
        self.max_beta = max(self.max_beta, abs(beta))
        self.max_yaw_rate = max(self.max_yaw_rate, abs(st["psi_dot_radps"]))
        stable = (abs(wrap(st["psi_rad"]-heading)) < sc.recovery_heading and
                  abs(beta) < sc.recovery_beta and
                  abs(st["psi_dot_radps"]-sc.road.curvature*st["v_x_mps"]) < sc.recovery_yaw_error)
        passed = bool(sc.obstacles) and all(s-o.s > b.length/2+o.length/2+5 for o in sc.obstacles)
        arrived = not sc.obstacles and s > sc.road.length-12
        stopped = math.hypot(st["v_x_mps"], st["v_y_mps"]) < 0.1
        # An unrevealed/behind hazard must not be counted as successfully addressed by a stop.
        relevant_stop = stopped and len(self.revealed) == len(sc.obstacles) and bool(sc.obstacles)
        if self.collision or self.offroad:
            self.recovery_since = None
            return
        # Latch hazard clearance separately from subsequent stability/recovery.
        if passed and self.passed_obstacles_at is None:
            self.passed_obstacles_at = b.time
        if relevant_stop and self.safe_stop_at is None:
            self.safe_stop_at = b.time
        recovered = stable and ((abs(d) < sc.recovery_lateral and (passed or arrived)) or relevant_stop)
        if recovered:
            if self.recovery_since is None:
                self.recovery_since = b.time
            if b.time-self.recovery_since >= sc.recovery_duration:
                self.status = "stopped_safe" if relevant_stop else "recovered"
        else:
            self.recovery_since = None

    def _record(self):
        b = self.backend
        st = b.state
        s, d, heading = self.scenario.road.project(st["x_m"], st["y_m"])
        row = {"t": b.time, **{k:v for k,v in st.items() if not isinstance(v,list)},
               "s": s, "d": d, "heading_error": wrap(st["psi_rad"]-heading),
               "beta_rad": math.atan2(st["v_y_mps"],max(abs(st["v_x_mps"]),0.2)),
               "steer_request_rad": b.command[0], "force_request_N": b.command[1],
               "force_executed_N": getattr(b,"executed_force",0.0), "abs_factor": b.abs_factor,
               "visible_obstacles": len(self.revealed), "min_road_margin_m": self.min_margin}
        row.update({"obstacles_passed": self.passed_obstacles_at is not None,
                    "safe_stop_achieved": self.safe_stop_at is not None,
                    "recovery_hold_s": 0.0 if self.recovery_since is None else b.time-self.recovery_since})
        for k in ("fx_N","fy_N","fz_N","kappa","alpha_rad","omega_radps","load_torque_Nm"):
            row.update({k+"_"+w: float(v) for w,v in zip(("fl","fr","rl","rr"),st[k])})
        row.update({"scale_true_"+w: v for w,v in zip(("fl","fr","rl","rr"),b.last_scale)})
        self.rows.append(row)

    def step(self, action):
        if self.status != "running":
            raise RuntimeError("Episode ended; call reset before step")
        self.backend.request(*action)
        for _ in range(round(self.scenario.control_dt/self.scenario.dt)):
            previous = self.backend.state
            previous_force = self.backend.force
            try:
                state = self.backend.step()
            except FloatingPointError:
                self.status = "numerical_failure"
                break
            self._check_geometry(previous, state)
            self._reveal()
            self._check_recovery()
            self.max_steer = max(self.max_steer,abs(state["steering_rad"]))
            self.max_steer_rate = max(self.max_steer_rate, abs(state["steering_rad"]-previous["steering_rad"])/self.scenario.dt)
            self.max_force_rate = max(self.max_force_rate,abs(self.backend.force-previous_force)/self.scenario.dt)
            self.min_actuator_force = min(self.min_actuator_force,self.backend.force)
            self.max_actuator_force = max(self.max_actuator_force,self.backend.force)
            if self.collision:
                self.status = "collision"
            elif self.offroad:
                self.status = "offroad"
            elif self.backend.time >= self.scenario.duration-1e-10 and self.status == "running":
                self.status = "timeout"
            if self.status != "running":
                break
        self.scale_history.append((self.backend.time,list(self.backend.last_scale)))
        if self.status != "numerical_failure":
            self._record()
        return self.observe(), self.status != "running", self.metrics()

    def metrics(self):
        b = self.backend
        s, d, heading = self.scenario.road.project(b.state["x_m"], b.state["y_m"])
        avoidance = (self.passed_obstacles_at is not None or self.safe_stop_at is not None) if self.scenario.obstacles else None
        recovered = self.status in ("recovered", "stopped_safe")
        # Allow only a small integration tolerance on the measured state derivative.
        actuator_violation = (self.max_steer > b.params["steering_actuator.angle_max_rad"]+1e-6
            or self.max_steer_rate > b.params["steering_actuator.angle_rate_max_radps"]+1e-3
            or self.max_force_rate > b.limits["force_rate_Nps"]+1e-6
            or self.min_actuator_force < -b.limits["max_brake_force_N"]-1e-6
            or self.max_actuator_force > b.limits["max_drive_force_N"]+1e-6)
        return {"status": self.status, "success": recovered and not actuator_violation,
                "avoidance_success": avoidance, "stable_recovery": recovered,
                "recovery_failed": bool(avoidance and not recovered and self.status != "running"),
                "passed_obstacles_at_s": self.passed_obstacles_at, "safe_stop_at_s": self.safe_stop_at,
                "collision": self.collision,"offroad":self.offroad,"simulation_time_s":b.time,
                "min_road_margin_m":self.min_margin,
                "min_sat_gap_m":self.min_gap if math.isfinite(self.min_gap) else None,
                "max_abs_beta_rad":self.max_beta,"max_abs_yaw_rate_radps":self.max_yaw_rate,
                "max_abs_steering_rad":self.max_steer,"max_abs_steering_rate_radps":self.max_steer_rate,
                "max_abs_force_rate_Nps":self.max_force_rate,"actuator_limit_violation":actuator_violation,
                "constraint_violation":bool(self.collision or self.offroad or actuator_violation or self.status=="numerical_failure"),
                "final_speed_mps":b.state["v_x_mps"],"final_s_m":s,
                "final_lateral_error_m":d, "final_heading_error_rad":wrap(b.state["psi_rad"]-heading),
                "recovery_hold_s":0 if self.recovery_since is None else b.time-self.recovery_since}
