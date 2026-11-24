"""Read-only artifact audit; write a new audit report next to the experiment directory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error("Output file already exists")
    root=args.directory
    episodes=json.loads((root/"episodes.json").read_text())
    sensitivity=json.loads((root/"sensitivity.json").read_text())
    mismatches=[];events=[];sources=set();valid_sources={};mirror=[];invalid_success=[]
    for row in episodes+sensitivity:
        folder=root/row["path"]
        manifest=json.loads((folder/"manifest.json").read_text())
        for filename,expected in manifest.items():
            p=folder/filename
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
                mismatches.append(str(p))
        config=json.loads((folder/"config.json").read_text())
        sources.add(json.dumps(config["implementation_sha256"],sort_keys=True))
        if row["success"] and row["constraint_violation"]:invalid_success.append(row["path"])
        if (folder/"plans.jsonl").exists():
            journal=[json.loads(line) for line in (folder/"plans.jsonl").read_text().splitlines()]
            if row in episodes:events+=journal
            for e in journal:
                selected=e["selected_index"]
                if selected is not None and e["candidates"][selected]["prediction"][0]<.5:
                    invalid_success.append(row["path"]+": selected infeasible candidate")
    for group in sorted({r["group_id"] for r in episodes}):
        l,r=[next(r for r in episodes if r["group_id"]==group and r["controller"]=="rolling" and r["preferred_side"]==s) for s in (1,-1)]
        with (root/l["path"]/"trace.csv").open() as f:a=list(csv.DictReader(f))
        with (root/r["path"]/"trace.csv").open() as f:b=list(csv.DictReader(f))
        same_length=len(a)==len(b)
        error=max((abs(float(x["y_m"])+float(y["y_m"])) for x,y in zip(a,b)),default=0.)
        mirror.append({"group_id":group,"same_outcome":l["status"]==r["status"],
                       "same_trace_length":same_length,"max_mirrored_y_error_m":error})
    latencies=[e["latency_ms"] for e in events]
    result={"episodes":len(episodes),"sensitivity_episodes":len(sensitivity),"hash_mismatches":mismatches,
        "implementation_versions":len(sources),"invalid_success_or_selection":invalid_success,
        "planning_calls":len(events),"planning_latency_ms":dict(zip(("p50","p95","p99","max"),
            np.percentile(latencies,[50,95,99,100]).tolist())),
        "planning_deadline_misses":sum(e["deadline_missed"] for e in events),
        "max_episode_control_20ms_deadline_miss_rate":max(r["controller_deadline_miss_rate"] for r in episodes),
        "actuator_limit_violations":sum(r["actuator_limit_violation"] for r in episodes),
        "mirror_checks":mirror,
        "all_checks_passed":not mismatches and not invalid_success and len(sources)==1
            and all(r["same_outcome"] and r["same_trace_length"] and r["max_mirrored_y_error_m"]<.01 for r in mirror)}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="mirror_checks"},indent=2))
    if not result["all_checks_passed"]:raise SystemExit(1)

if __name__=="__main__":main()
