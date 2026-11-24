"""Physical gates for internal friction and independent predictor error, without fitting test outcomes."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extreme_avoidance.backend import OCDBackend
from extreme_avoidance.scenario import Scenario,Road,Patch
from extreme_avoidance.planner import predictor_parameters
from extreme_avoidance._ocd_native import predict_actions
from validate_physics import simulate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sc=Scenario(name="internal_mu_validation",friction_mode="internal_mu",obstacles=(),road=Road(width=1000,length=1000))
    checks=[];errors=[]
    def check(name,ok,values):
        checks.append({"name":name,"passed":bool(ok),"measurements":values})
        print(name,"PASS" if ok else "FAIL",values,flush=True)
    distances=[]
    for scale in (1.,.6,.5):
        arr,b=simulate(replace(sc,speed=20,base_scale=scale,duration=8),lambda b:(0,-8000),args.output/f"stop_{scale}.csv")
        idx=np.flatnonzero(abs(arr["v_x_mps"])<.1)
        d=float(arr["x_m"][idx[0]]) if len(idx) else None
        distances.append(d)
        check(f"internal_stop_{scale}",d is not None and np.max(abs(arr["v_x_mps"][-25:]))<.1,
              {"stopping_distance_m":d,"final_speed_mps":float(arr["v_x_mps"][-1])})
    check("reduced_capability_increases_stop_distance",all(x is not None for x in distances) and distances==sorted(distances),{"distances_m":distances})
    left,_=simulate(replace(sc,speed=15,base_scale=.6,duration=2),lambda b:(.04,0),args.output/"left.csv")
    right,_=simulate(replace(sc,speed=15,base_scale=.6,duration=2),lambda b:(-.04,0),args.output/"right.csv")
    mirror=max(float(np.max(abs(left[k]+right[k]))) for k in ("y_m","psi_rad","psi_dot_radps"))
    check("internal_mirror",mirror<1e-5,{"max_mirror_error":mirror})
    full,_=simulate(replace(sc,speed=20,base_scale=.6,duration=3),lambda b:(.04,-2400),args.output/"combined_1ms.csv")
    half,_=simulate(replace(sc,speed=20,base_scale=.6,duration=3,dt=.0005),lambda b:(.04,-2400),args.output/"combined_05ms.csv")
    xy=float(np.max(np.hypot(full["x_m"]-half["x_m"],full["y_m"]-half["y_m"])))
    check("internal_step_halving",xy<.1,{"position_difference_m":xy})
    patch,_=simulate(replace(sc,speed=15,duration=3,patches=(Patch(20,100,scale=.5),)),lambda b:(0,0),args.output/"patch.csv")
    entry={w:float(patch["t"][np.flatnonzero(patch["scale_"+w]<1)[0]]) for w in ("fl","fr","rl","rr")}
    check("internal_wheel_local_patch",entry["fl"]==entry["fr"] and entry["fl"]<entry["rl"]==entry["rr"],entry)
    split,_=simulate(replace(sc,speed=15,duration=3,patches=(Patch(0,100,d_min=0,scale=.5),)),
                     lambda b:(0,-6000),args.output/"split_brake.csv")
    yaw=float(max(abs(split["psi_dot_radps"])))
    check("internal_split_braking_yaw",yaw>.01,{"max_yaw_rate_radps":yaw})
    # Open-loop comparison on a frozen sweep. Report errors, do not call this model identification.
    for speed in (15.,20.):
        for scale in (.5,1.):
            for steer,force in ((0.,-2400.),(.02,0.),(.04,-1200.)):
                tag=f"prediction_v{speed}_g{scale}_d{steer}_f{force}"
                scenario=replace(sc,speed=speed,base_scale=scale,duration=5)
                truth,b=simulate(scenario,lambda b:(steer,force),args.output/(tag+".csv"))
                actions=np.tile([steer,force],(500,1));actions[:4]=0
                predicted=predict_actions([0,0,0,speed,0,0,0,0],actions,.01,predictor_parameters(b,scale))
                predicted=predicted[2::2]
                for horizon in (1.,2.,5.):
                    n=int(horizon/.02)
                    xy=np.hypot(predicted[:n,0]-truth["x_m"][:n],predicted[:n,1]-truth["y_m"][:n])
                    errors.append({"speed":speed,"scale":scale,"steer":steer,"force":force,"horizon_s":horizon,
                        "max_position_error_m":float(max(xy)),
                        "max_lateral_error_m":float(max(abs(predicted[:n,1]-truth["y_m"][:n]))),
                        "max_speed_error_mps":float(max(abs(predicted[:n,3]-truth["v_x_mps"][:n]))),
                        "max_yaw_error_rad":float(max(abs(predicted[:n,2]-truth["psi_rad"][:n])))})
                np.savez_compressed(args.output/(tag+".npz"),prediction=predicted,
                                    truth=np.stack([truth[k] for k in ("x_m","y_m","psi_rad","v_x_mps")],axis=1))
    report={"checks":checks,"passed":sum(x["passed"] for x in checks),"total":len(checks),
        "predictor_errors":errors,"prediction_error_claim":"measured on 12 open-loop input conditions over 1/2/5 s; not a guaranteed error bound",
        "vehicle_sha256":hashlib.sha256((ROOT/"configs/vehicle.json").read_bytes()).hexdigest(),
        "planner_sha256":hashlib.sha256((ROOT/"configs/planner.json").read_bytes()).hexdigest()}
    report["implementation_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),*(ROOT/"native").glob("*.hpp"),*(ROOT/"native").glob("*.cpp"),
                  *(ROOT/"extreme_avoidance").glob("*.so")]}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("predictor worst errors",{h:max(e["max_position_error_m"] for e in errors if e["horizon_s"]==h) for h in (1.,2.,5.)})
    if report["passed"]!=report["total"]:raise SystemExit(1)

if __name__=="__main__":main()
