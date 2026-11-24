"""Receding-horizon feedback-maneuver lattice; no learned model or future truth access."""
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
from .scenario import ROOT
from .backend import PREFIX
from .controllers import Baseline
from ._ocd_native import TireBench, maneuver_action, screen_maneuvers


def predictor_parameters(vehicle, scale=1.0, mode="internal_mu", config=None):
    """Build from frozen vehicle parameters and pure-slip reference calibration."""
    cfg = config or json.loads((ROOT/"configs/planner.json").read_text())
    p = vehicle.params
    mass, lf, lr = vehicle.mass, vehicle.lf, vehicle.lr
    stiffness = []
    for wheel, load in (("front_left", mass*9.81*lr/(lf+lr)/2),
                        ("rear_left", mass*9.81*lf/(lf+lr)/2)):
        prefix = PREFIX+"z.tire_model."+wheel+"."
        tire = TireBench({k[len(prefix):]: v for k,v in p.items() if k.startswith(prefix)})
        eps = 1e-5
        stiffness.append(2*(tire.forces(0,eps,load,1)[1]-tire.forces(0,-eps,load,1)[1])/(2*eps))
    if mode == "post_force":
        stiffness = [c*scale for c in stiffness]
    return [mass, p[PREFIX+"moment_of_inertia_kgpm2.z"], lf, lr,
            p[PREFIX+"cog.height_m"], *stiffness,
            cfg["effective_mu_x_at_scale_1"]*scale, cfg["effective_mu_y_at_scale_1"]*scale,
            p["steering_actuator.T_PT1"], vehicle.limits["force_time_constant_s"],
            p["steering_actuator.angle_max_rad"], p["steering_actuator.angle_rate_max_radps"],
            vehicle.limits["force_rate_Nps"], vehicle.limits["max_brake_force_N"],
            vehicle.limits["max_drive_force_N"], p["drivetrain.brake_bias_front"]]


def state_vector(obs):
    s=obs["state"]
    return [s[k] for k in ("x_m","y_m","psi_rad","v_x_mps","v_y_mps","psi_dot_radps","steering_rad")]+[obs["actuator_force_N"]]


class RollingPlanner:
    def __init__(self, road, vehicle, speed, control_dt, side=1, config=None):
        if abs(road.curvature)>1e-12:
            raise ValueError("Rolling planner v1 supports straight roads only")
        self.config=dict(config or json.loads((ROOT/"configs/planner.json").read_text()))
        c=self.config
        if not 0<c["prediction_dt_s"]<=.02 or not 0<c["horizon_s"]<=12:
            raise ValueError("Invalid prediction time grid")
        if not 0<c["deadline_s"]<=c["replan_period_s"] or c["geometry_margin_m"]<0:
            raise ValueError("Invalid planner deadline/margin")
        if abs(round(c["replan_period_s"]/control_dt)*control_dt-c["replan_period_s"])>1e-8:
            raise ValueError("Replanning period must be an integer multiple of control_dt")
        if side not in (-1,1): raise ValueError("side must be -1 or 1")
        self.base_parameters=predictor_parameters(vehicle,config=c)
        self.length,self.width,self.road_width=vehicle.length,vehicle.width,road.width
        self.delay=vehicle.limits["command_delay_s"]
        if abs(round(self.delay/c["prediction_dt_s"])*c["prediction_dt_s"]-self.delay)>1e-8:
            raise ValueError("Command delay must align with prediction grid")
        self.speed,self.side=speed,side
        self.control_dt=control_dt
        self.cruise=Baseline("cruise",road,vehicle,speed)
        self.aeb=Baseline("aeb",road,vehicle,speed)
        self.trigger=None
        self.selected=None
        self.obstacle=None
        self.next_plan=0.0
        self.journal=[]
        self.last_event={"planner_replanned":False,"planner_fallback":False}

    def _parameters(self,obs):
        scale=min(obs["capability_scale_estimate"])
        if not math.isfinite(scale) or scale<=0: raise ValueError("Invalid capability estimate")
        p=list(self.base_parameters)
        p[7]*=scale;p[8]*=scale
        if obs["friction_mode"]=="post_force":p[5]*=scale;p[6]*=scale
        return p

    def _candidates(self):
        ob=self.obstacle;c=self.config
        ramp=max(5.,ob.s-ob.length/2-self.length/2-2-self.trigger)
        result=[]
        if self.selected is not None:result.append(list(self.selected))
        for side,offset,ramp_factor,brake,release in itertools.product(
                (self.side,-self.side),c["offsets_m"],c["ramp_factors"],
                c["brake_forces_N"],c["brake_end_offsets_m"]):
            result.append([side*offset,ramp*ramp_factor,brake,ob.s+release,c["return_length_m"],0.])
        result.append([0.,ramp,0.,ob.s,c["return_length_m"],1.])
        return list(dict.fromkeys(tuple(z) for z in result))

    def _delay_queue(self,obs):
        queue=[]
        for k in range(round(self.delay/self.config["prediction_dt_s"])):
            action=obs["applied_command"]
            for due,cmd in obs["pending_commands"]:
                if due<=obs["time"]+k*self.config["prediction_dt_s"]+1e-10:action=cmd
            queue.append(action)
        return queue

    def __call__(self,obs):
        started=time.perf_counter()
        self.last_event={"planner_replanned":False,"planner_fallback":self.selected is None,
                         "planner_selected_offset_m":0. if self.selected is None else self.selected[0]}
        if not obs["obstacles"]:
            self.last_event["planner_fallback"]=False
            return self.cruise(obs)
        if self.trigger is None:
            self.trigger=obs["s"]
            self.obstacle=min(obs["obstacles"],key=lambda o:o.s)
        p=self._parameters(obs)
        x=state_vector(obs)
        if obs["time"]+1e-9>=self.next_plan:
            candidates=self._candidates()
            obs_list=sorted(obs["obstacles"],key=lambda o:o.s)
            obstacles=[[o.s,o.d,o.length,o.width] for o in obs_list]
            scores=screen_maneuvers(x,candidates,obstacles,self._delay_queue(obs),self.trigger,self.speed,
                self.road_width,self.length,self.width,self.config["horizon_s"],
                self.config["prediction_dt_s"],self.control_dt,self.config["geometry_margin_m"],p)
            adjusted=scores[:,1].copy()
            if self.selected is not None:
                adjusted+=self.config["switch_penalty"]*np.array([(z[0]-self.selected[0])**2 for z in candidates])
            valid=np.flatnonzero(scores[:,0]>.5)
            selected=int(valid[np.argmin(adjusted[valid])]) if len(valid) else None
            proposed=None if selected is None else candidates[selected]
            action=self.aeb(obs) if proposed is None else maneuver_action(x,proposed,self.trigger,
                self.obstacle.s,self.obstacle.length,self.length,self.speed,p)
            # Timing includes candidates, model rollout, checks, ranking and action conversion.
            elapsed=time.perf_counter()-started
            late=elapsed>self.config["deadline_s"]
            if late:
                proposed=None;action=self.aeb(obs)
            self.selected=proposed
            self.next_plan=obs["time"]+self.config["replan_period_s"]
            event={"t":obs["time"],"latency_ms":elapsed*1000,"deadline_missed":late,
                   "fallback":proposed is None,"selected_index":selected if not late else None,
                   "candidate_count":len(candidates),"predicted_feasible_count":int(len(valid)),
                   "effective_mu_x":p[7],"effective_mu_y":p[8],
                   "candidates":[{"parameters":z,"prediction":row.tolist()} for z,row in zip(candidates,scores)]}
            self.journal.append(event)
            self.last_event={"planner_replanned":True,"planner_fallback":proposed is None,
                             "planner_selected_offset_m":0. if proposed is None else proposed[0],
                             "planner_latency_ms":elapsed*1000}
            return action
        if self.selected is None:return self.aeb(obs)
        return maneuver_action(x,self.selected,self.trigger,self.obstacle.s,self.obstacle.length,
                               self.length,self.speed,p)

    def metrics(self):
        events=self.journal
        latencies=[e["latency_ms"] for e in events]
        return {"planner_calls":len(events),"planner_p95_ms":float(np.percentile(latencies,95)) if events else 0.,
                "planner_p99_ms":float(np.percentile(latencies,99)) if events else 0.,
                "planner_max_ms":max(latencies,default=0.),
                "planner_deadline_miss_rate":sum(e["deadline_missed"] for e in events)/max(len(events),1),
                "planner_fallback_rate":sum(e["fallback"] for e in events)/max(len(events),1),
                "planner_prediction_scope":"candidate feasibility in approximate bicycle, not OCD verification",
                "planner_timing_scope":"candidate generation, rollout, swept checks, ranking, action; excludes disk logging and OCD stepping"}
