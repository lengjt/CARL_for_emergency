#!/usr/bin/env python3
"""Reproducible numerical/physical sanity checks, not real-vehicle validation."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from extreme_avoidance.backend import OCDBackend, PREFIX
from extreme_avoidance.scenario import Scenario, Road, Patch
from extreme_avoidance._ocd_native import TireBench


def simulate(sc, command, output):
    b = OCDBackend(sc)
    rows = []
    steps = round(sc.duration / sc.dt)
    period = round(sc.control_dt / sc.dt)
    for i in range(steps):
        if i % period == 0:
            b.request(*command(b))
        b.step()
        if (i + 1) % period == 0:
            row = {"t": b.time}
            for k, v in b.state.items():
                if isinstance(v, list):
                    row.update({k+"_"+w: x for w, x in zip(("fl","fr","rl","rr"),v)})
                else:
                    row[k] = v
            row.update({"scale_"+w: x for w,x in zip(("fl","fr","rl","rr"),b.last_scale)})
            row.update({"force_request_N": b.command[1], "force_executed_N": b.executed_force,
                        "abs_factor": b.abs_factor})
            rows.append(row)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, passed, measurements, criterion):
        checks.append(dict(name=name, passed=bool(passed), measurements=measurements, criterion=criterion))
        print(name, "PASS" if passed else "FAIL", measurements, flush=True)

    manifest = json.loads((ROOT/"upstream/manifest.json").read_text())
    bad = []
    for entry in manifest["files"]:
        path = ROOT/entry["dest"]
        if not path.is_file():
            bad.append(entry["dest"]); continue
        content = path.read_bytes()
        actual = hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
        if actual != entry["sha"]:
            bad.append(entry["dest"])
    check("upstream_source_integrity", not bad, {"files":len(manifest["files"]),"mismatches":bad},
          "Every selected source matches its recorded Git blob ID")

    sc = Scenario(name="physics", speed=15, duration=6, obstacles=(), road=Road(width=1000,length=1000))
    static,b = simulate(replace(sc,speed=0,duration=2),lambda b:(0,0),args.output/"static.csv")
    weight = b.mass*9.81
    load = sum(static["fz_N_"+w][-1] for w in ("fl","fr","rl","rr"))
    check("static_equilibrium", abs(load/weight-1)<0.01 and abs(static["v_x_mps"][-1])<1e-6,
          {"sum_load_N":float(load),"weight_N":weight,"speed_mps":float(static["v_x_mps"][-1])},
          "Static total vertical load within 1% of mg and speed below 1e-6 m/s")

    coast,_ = simulate(sc,lambda b:(0,0),args.output/"coast.csv")
    check("coast_and_straight_symmetry", coast["v_x_mps"][-1]<coast["v_x_mps"][0] and
          max(abs(coast["y_m"]))<1e-5 and max(abs(coast["psi_rad"]))<1e-5,
          {"final_speed_mps":float(coast["v_x_mps"][-1]),"max_lateral_m":float(max(abs(coast["y_m"])))},
          "Unpowered car decelerates and remains straight within 1e-5 m/rad")

    brakes = {}
    for scale in (1.0,0.5):
        a,_ = simulate(replace(sc,speed=20,base_scale=scale),lambda b:(0,-8000),
                       args.output/f"brake_{scale}.csv")
        stopped = np.flatnonzero(abs(a["v_x_mps"])<0.1)
        distance = float(a["x_m"][stopped[0]]) if len(stopped) else None
        brakes[scale] = distance
        check(f"brake_stop_scale_{scale}", distance is not None and np.max(abs(a["v_x_mps"][-25:]))<0.1,
              {"stopping_distance_m":distance,"residual_speed_mps":float(a["v_x_mps"][-1])},
              "Reach <0.1 m/s within 6 s and stay <0.1 m/s for last 0.5 s; no forced zeroing")
    check("reduced_grip_stopping_distance", all(v is not None for v in brakes.values()) and brakes[0.5]>brakes[1.0],
          {str(k):v for k,v in brakes.items()},"Halving tire-force scale increases stopping distance")

    def turn(side, brake=False):
        def command(b):
            force = -2500 if brake and b.time>=1 else b.mass*1.5*(15-b.state["v_x_mps"])+334
            return side*0.035,force
        return command
    left,_ = simulate(sc,turn(1),args.output/"turn_left.csv")
    right,_ = simulate(sc,turn(-1),args.output/"turn_right.csv")
    mirror = max(float(np.max(abs(left[k]+right[k]))) for k in ("y_m","psi_rad","v_y_mps","psi_dot_radps"))
    check("left_right_mirror", mirror<0.05, {"max_mirror_error_m_rad_mps_radps":mirror},
          "Mirrored steering gives sign-reversed lateral states within 0.05 (component SI units)")
    lateral_accel = left["ay_mps2"][-50:]
    expected = left["v_x_mps"][-50:]*left["psi_dot_radps"][-50:]
    error = float(np.max(abs(lateral_accel-expected)))
    yaw_std = float(np.std(left["psi_dot_radps"][-50:]))
    check("steady_turn_balance", error<0.1 and yaw_std<0.01,
          {"max_ay_minus_vx_r_mps2":error,"yaw_rate_std_radps":yaw_std},
          "Final 1 s: ay ~= vx*r within 0.1 m/s2 and yaw-rate std <0.01 rad/s")
    combined,_ = simulate(replace(sc,duration=3),turn(1,True),args.output/"turn_brake.csv")
    check("turn_and_brake", combined["v_x_mps"][-1]<combined["v_x_mps"][49] and
          all(np.all(np.isfinite(v)) for v in combined.values()) and np.max(abs(combined["psi_dot_radps"]))>0.01,
          {"final_speed_mps":float(combined["v_x_mps"][-1]),"max_yaw_rate_radps":float(max(abs(combined["psi_dot_radps"])))},
          "Combined input remains finite, decelerates and produces yaw response")

    half,_ = simulate(replace(sc,dt=0.0005),turn(1),args.output/"turn_half_step.csv")
    xy_error = float(np.max(np.hypot(left["x_m"]-half["x_m"],left["y_m"]-half["y_m"])))
    vx_error = float(np.max(abs(left["v_x_mps"]-half["v_x_mps"])))
    yaw_error = float(np.max(abs(left["psi_rad"]-half["psi_rad"])))
    check("step_halving", xy_error<0.1 and vx_error<0.1 and yaw_error<0.02,
          {"max_position_m":xy_error,"max_speed_mps":vx_error,"max_yaw_rad":yaw_error},
          "1 ms vs 0.5 ms at identical 20 ms command times: position <0.1 m, vx <0.1 m/s, yaw <0.02 rad")

    patch_sc = replace(sc,duration=3,patches=(Patch(s_start=20,s_end=100,scale=0.5),))
    patch,_ = simulate(patch_sc,lambda b:(0,0),args.output/"patch_entry.csv")
    entry = {w:float(patch["t"][np.flatnonzero(patch["scale_"+w]<1)[0]]) for w in ("fl","fr","rl","rr")}
    check("wheel_local_grip", entry["fl"]==entry["fr"] and entry["fl"]<entry["rl"] and entry["rl"]==entry["rr"],
          entry,"Front axle enters spatial patch before rear axle, with simultaneous left/right entry")
    split_sc = replace(sc,duration=4,patches=(Patch(s_start=0,s_end=100,d_min=0,scale=0.5),))
    split,_ = simulate(split_sc,lambda b:(0,-6000),args.output/"split_brake.csv")
    check("split_grip_braking_yaw", np.max(abs(split["psi_dot_radps"]))>0.01,
          {"max_yaw_rate_radps":float(max(abs(split["psi_dot_radps"])))},
          "Left/right unequal tire-force scale under straight braking induces yaw >0.01 rad/s")

    prefix = PREFIX+"z.tire_model.front_left."
    tire = TireBench({k[len(prefix):]:v for k,v in b.params.items() if k.startswith(prefix)})
    tire_rows = []
    curves = []
    for fz in (1500,2500,3500):
        for scale in (1.0,0.6,0.5):
            fx = [tire.forces(float(k),0,fz,scale)[0] for k in np.linspace(-0.3,0.3,301)]
            fy = [tire.forces(0,float(a),fz,scale)[1] for a in np.linspace(-0.25,0.25,301)]
            tire_rows.append({"fz_N":fz,"scale":scale,"peak_abs_Fx_over_Fz":max(abs(np.array(fx)))/fz,
                              "peak_abs_Fy_over_Fz":max(abs(np.array(fy)))/fz})
            curves.append({"fz_N":fz,"scale":scale,"fx_N":fx,"fy_N":fy})
    (args.output/"tire_calibration.json").write_text(json.dumps({"scope":"front-left tire; finite sampled slip grid, not road mu identification",
        "kappa_range":[-0.3,0.3,301],"alpha_range_rad":[-0.25,0.25,301],"peaks":tire_rows,"curves":curves},indent=2)+"\n")
    pure_x=tire.forces(.1,0,2500,1)[0]; pure_y=tire.forces(0,.1,2500,1)[1]
    combined_force=tire.forces(.1,.1,2500,1)
    scaled=tire.forces(.1,.1,2500,.5)
    check("combined_slip_and_scale", abs(combined_force[0])<abs(pure_x) and abs(combined_force[1])<abs(pure_y)
          and np.allclose(scaled,np.array(combined_force)*.5),
          {"pure_Fx_N":pure_x,"pure_Fy_N":pure_y,"combined_N":combined_force},
          "At kappa=0.1, alpha=0.1 rad: combined-slip reduces both forces; scale=0.5 halves them")

    report={"scope":"Engineering sanity checks on a synthetic 800 kg research car; not experimental validation",
            "checks":checks,"passed":sum(c["passed"] for c in checks),"total":len(checks),
            "vehicle_sha256":hashlib.sha256((ROOT/"configs/vehicle.json").read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "upstream_commit":b.commit,"scenario":sc.to_dict()}
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    sys.exit(0 if report["passed"]==report["total"] else 1)

if __name__ == "__main__":
    main()
