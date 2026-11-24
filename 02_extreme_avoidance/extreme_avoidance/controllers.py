"""Transparent baselines: AEB, steering-only response, and joint steering/braking.

The maneuver is a quintic lateral offset in road coordinates, tracked in feedback.
This is not a reproduced TAMP algorithm, an MPC, or an optimal expert.
"""
import math
import numpy as np
from .scenario import wrap

def smoothstep(u):
    u = float(np.clip(u,0,1))
    return 10*u**3-15*u**4+6*u**5, 30*u**2-60*u**3+30*u**4, 60*u-180*u**2+120*u**3

class Baseline:
    def __init__(self, mode, road, vehicle, speed, side=1, offset=2.8, brake=3000.0, ramp_length=None):
        if mode not in ("aeb","steer","joint","joint_legacy","cruise") or side not in (-1,1):
            raise ValueError("Invalid controller mode/side")
        if not all(math.isfinite(x) for x in (speed, offset, brake)) or min(speed,offset,brake) < 0:
            raise ValueError("Controller parameters must be finite and nonnegative")
        if ramp_length is not None and (not math.isfinite(ramp_length) or ramp_length <= 0):
            raise ValueError("ramp_length must be finite and positive")
        self.mode,self.road,self.vehicle,self.speed = mode,road,vehicle,speed
        self.side,self.offset,self.brake,self.ramp_length = side,offset,brake,ramp_length
        self.trigger_s = None
        self.obstacle = None

    def reference(self,s):
        if self.trigger_s is None or self.mode not in ("steer","joint","joint_legacy"):
            return 0.0,0.0,0.0
        start = self.trigger_s
        end = self.obstacle.s-self.obstacle.length/2-self.vehicle.length/2-2
        length = max(5.0,end-start) if self.ramp_length is None else self.ramp_length
        return_start = self.obstacle.s+self.obstacle.length/2+self.vehicle.length/2+5
        return_length = 35.0
        a, da, dda = smoothstep((s-start)/length)
        b, db, ddb = smoothstep((s-return_start)/return_length)
        amp = self.side*self.offset
        return amp*(a-b),amp*(da/length-db/return_length),amp*(dda/length**2-ddb/return_length**2)

    def __call__(self, obs):
        st = obs["state"]
        if self.trigger_s is None and obs["obstacles"]:
            self.trigger_s = obs["s"]
            self.obstacle = min(obs["obstacles"],key=lambda o:o.s)
        d_ref,d_prime,d_second = self.reference(obs["s"])
        yaw_ref = obs["road_heading"]+math.atan2(d_prime,1-self.road.curvature*d_ref)
        curvature = self.road.curvature + d_second/(1+d_prime*d_prime)**1.5
        speed = max(st["v_x_mps"],0.0)
        beta = math.atan2(st["v_y_mps"],max(speed,1))
        steer = (math.atan(self.vehicle.wheelbase*curvature)
                 +0.9*wrap(yaw_ref-st["psi_rad"])
                 +math.atan2(1.2*(d_ref-obs["d"]),speed+3)
                 -0.25*beta +0.10*(speed*curvature-st["psi_dot_radps"]))
        # Maintain pre-event cruise, including aero and rolling resistance compensation.
        force = self.vehicle.mass*1.5*(self.speed-speed)+0.025*self.vehicle.mass*9.81+0.6125*speed*speed
        if self.trigger_s is not None:
            if self.mode == "aeb":
                force = -self.vehicle.limits["max_brake_force_N"]
            elif self.mode == "steer":
                force = 0.0  # coast while steering; no acceleration to mask speed loss
            elif self.mode in ("joint", "joint_legacy") and obs["s"] < self.obstacle.s+5:
                # Prevent stopping mid-swerve; hold a modest crawl if necessary.
                force = -self.brake if speed > 6.0 else self.vehicle.mass*(6.0-speed)
            if self.mode == "joint" and force > 0:
                # Positive traction is delayed until the entire lane-return maneuver settles.
                settled = (obs["s"] > self.obstacle.s+self.obstacle.length/2+self.vehicle.length/2+5
                           and abs(obs["d"]) < .35 and abs(beta) < .04
                           and abs(wrap(st["psi_rad"]-obs["road_heading"])) < .05
                           and abs(st["psi_dot_radps"]-speed*self.road.curvature) < .08)
                force = min(force, 1000.0) if settled else 0.0
        return float(steer),float(force)
