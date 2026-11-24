import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from .scenario import Scenario, ROOT
from .experiment import run, save

def main():
    parser=argparse.ArgumentParser(description="Python + Open Car Dynamics emergency avoidance")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("run")
    p.add_argument("--scenario",default="straight_popup")
    p.add_argument("--controller",choices=("aeb","steer","joint","joint_legacy","cruise","rolling"),default="joint")
    p.add_argument("--side",type=int,choices=(-1,1),default=1)
    p.add_argument("--brake",type=float,default=3000)
    p.add_argument("--offset",type=float,default=2.8)
    p.add_argument("--dt",type=float)
    p.add_argument("--friction-mode",choices=("post_force","internal_mu"))
    p.add_argument("--planner-config",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--vehicle",type=Path)
    p.add_argument("--no-visual",action="store_true")
    p=sub.add_parser("sample",help="offline closed-loop template bank; retains every trial")
    p.add_argument("--scenario",default="straight_critical")
    p.add_argument("--output",type=Path)
    args=parser.parse_args()
    sc=Scenario.load(args.scenario)
    if args.command=="run":
        if args.dt is not None: sc=replace(sc,dt=args.dt)
        if args.friction_mode is not None:sc=replace(sc,friction_mode=args.friction_mode)
        output=args.output or ROOT/"artifacts"/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+"_"+sc.name+"_"+args.controller)
        if output.exists(): parser.error("Output directory already exists; choose a new directory")
        planner_config=json.loads(args.planner_config.read_text()) if args.planner_config else None
        env,metrics,meta=run(sc,args.controller,args.side,args.offset,args.brake,vehicle_file=args.vehicle,
                             planner_config=planner_config)
        save(env,metrics,meta,output,not args.no_visual)
        print(json.dumps({"output":str(output),**metrics},indent=2))
    else:
        output=args.output or ROOT/"artifacts"/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+"_samples")
        if output.exists(): parser.error("Sample output directory already exists")
        output.mkdir(parents=True)
        index=[]
        for side in (-1,1):
            for offset in (2.6,3.0):
                for brake in (0.,1500.,3000.):
                    mode="steer" if brake==0 else "joint"
                    env,metrics,meta=run(sc,mode,side,offset,brake)
                    tag=f"trial_{len(index):03d}"
                    save(env,metrics,meta,output/tag,visual=False)
                    item={"path":tag,"controller":meta["controller"],**metrics}
                    index.append(item)
                    (output/"index.json").write_text(json.dumps(index,indent=2)+"\n")
                    print(tag,metrics["status"],flush=True)
        summary={"trials":len(index),"valid":sum(r["success"] for r in index),
                 "valid_sides":sorted(set(r["controller"]["side"] for r in index if r["success"])),
                 "feasibility_label":"verified_feasible" if any(r["success"] for r in index) else "unknown",
                 "label_scope":"validity in this synthetic vehicle/scenario and finite recovery horizon; not global optimality",
                 "wall_time_s":sum(r["wall_time_s"] for r in index)}
        (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
