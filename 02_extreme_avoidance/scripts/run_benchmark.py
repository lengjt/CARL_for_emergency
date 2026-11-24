"""Frozen finite-grid comparison, retaining every episode and planning decision."""
import argparse
from dataclasses import replace
import hashlib
import html
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extreme_avoidance.scenario import Scenario,Road,Obstacle
from extreme_avoidance.experiment import run,save


def write_json(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/benchmark.json")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.config.read_text());planner=json.loads((ROOT/"configs/planner.json").read_text())
    manifest={"benchmark":cfg,"planner":planner,"created_local_date":"2026-10-15",
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder,pattern in (("extreme_avoidance","*.py"),("native","*.hpp"),("native","*.cpp"),("scripts","run_benchmark.py"))
            for p in (ROOT/folder).glob(pattern)}}
    write_json(args.output/"benchmark_manifest.json",manifest)
    results=[];scenario_index=[];started=time.perf_counter()
    for speed,scale,reveal in itertools.product(cfg["speeds_mps"],cfg["capability_scales"],cfg["reveal_distances_m"]):
        group=f"v{speed:g}_g{scale:g}_d{reveal:g}"
        sc=Scenario(name=group,speed=speed,base_scale=scale,friction_mode=cfg["friction_mode"],
            duration=cfg["duration_s"],dt=cfg["physics_dt_s"],control_dt=cfg["control_dt_s"],
            road=Road(width=cfg["road_width_m"],length=cfg["road_length_m"]),
            obstacles=(Obstacle(s=cfg["obstacle_s_m"],reveal_distance=reveal),))
        group_results=[]
        for mode in cfg["controllers"]:
            for side in (cfg["rolling_preferred_sides"] if mode=="rolling" else [1]):
                tag=group+"_"+mode+("_left" if side==1 else "_right")
                env,metrics,meta=run(sc,mode,side=side,planner_config=planner)
                meta["group_id"]=group
                save(env,metrics,meta,args.output/tag,visual="replay")
                item={"path":tag,"group_id":group,"speed_mps":speed,"scale":scale,
                      "reveal_distance_m":reveal,"controller":mode,"preferred_side":side,**metrics}
                group_results.append(item);results.append(item)
                write_json(args.output/"episodes.json",results)
                print(tag,metrics["status"],"clear",metrics["avoidance_success"],"recover",metrics["stable_recovery"],
                      "p95_ms",round(metrics.get("planner_p95_ms",0),2),flush=True)
        aeb=next(r for r in group_results if r["controller"]=="aeb")
        any_success=any(r["success"] for r in group_results)
        category=("aeb_sufficient" if aeb["success"] else "verified_non_aeb_solution" if any_success
                  else "clearance_without_recovery" if any(r["avoidance_success"] for r in group_results)
                  else "feasibility_unknown")
        scenario_index.append({"group_id":group,"scenario":sc.to_dict(),"category":category,
            "feasibility":"verified_feasible" if any_success else "unknown",
            "valid_episode_paths":[r["path"] for r in group_results if r["success"]],
            "interpretation":"AEB failure does not prove steering necessary or physical infeasibility"})
        write_json(args.output/"scenarios.json",scenario_index)
    # Independent model/step sensitivity on the pre-existing development contrast case.
    sensitivity=[]
    base=replace(Scenario.load("straight_critical"),friction_mode="internal_mu")
    variants=[("reference",base,"rolling"),("half_step",replace(base,dt=.0005),"rolling"),
              ("post_force",replace(base,friction_mode="post_force"),"rolling"),
              ("estimate_low",replace(base,scale_estimate_factor=.8),"rolling"),
              ("estimate_high",replace(base,scale_estimate_factor=1.2),"rolling"),
              ("recovery_old",Scenario.load("straight_low_grip"),"joint_legacy"),
              ("recovery_fixed",Scenario.load("straight_low_grip"),"joint")]
    for name,sc,mode in variants:
        env,m,meta=run(sc,mode,planner_config=planner)
        save(env,m,meta,args.output/("check_"+name),visual=True)
        sensitivity.append({"variant":name,"path":"check_"+name,**m})
        print("check",name,m["status"],flush=True)
    write_json(args.output/"sensitivity.json",sensitivity)
    reference=next(r for r in sensitivity if r["variant"]=="reference")
    half=next(r for r in sensitivity if r["variant"]=="half_step")
    comparison={"same_outcome":reference["status"]==half["status"],"status_1ms":reference["status"],
                "status_05ms":half["status"],"final_s_difference_m":half["final_s_m"]-reference["final_s_m"]}
    write_json(args.output/"step_comparison.json",comparison)
    summary={"scope":cfg["scope"],"scenario_groups":len(scenario_index),"episodes":len(results),
             "wall_time_s":time.perf_counter()-started,"by_controller":{}}
    for mode in cfg["controllers"]:
        rows=[r for r in results if r["controller"]==mode]
        summary["by_controller"][mode]={"episodes":len(rows),"avoidance_successes":sum(r["avoidance_success"] for r in rows),
            "stable_recoveries":sum(r["stable_recovery"] for r in rows),
            "valid_successes":sum(r["success"] for r in rows),
            "avoidance_rate":sum(r["avoidance_success"] for r in rows)/len(rows),
            "recovery_rate":sum(r["stable_recovery"] for r in rows)/len(rows),
            "recovery_failures":sum(r["recovery_failed"] for r in rows),
            "constraint_violations":sum(r["constraint_violation"] for r in rows),
            "max_controller_p99_ms":max(r["controller_p99_ms"] for r in rows),
            "max_episode_planner_p99_ms":max(r.get("planner_p99_ms",0) for r in rows),
            "max_episode_planner_deadline_miss_rate":max(r.get("planner_deadline_miss_rate",0) for r in rows)}
    write_json(args.output/"summary.json",summary)
    categories={r["group_id"]:r["category"] for r in scenario_index}
    header='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>在线避障规划基线</title>
<style>body{font:15px system-ui;margin:30px auto;max-width:1400px;color:#20314a;background:#f5f7fa;padding:20px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:9px;border-bottom:1px solid #dce3eb;text-align:left}a{color:#1460b5}.ok{color:#176c46}.bad{color:#b73434}</style>
<h1>滚动采样规划 · 直路避障实验集</h1><p>18 个基础场景：速度 15/20/25 m/s × 内部附着倍率 0.5/0.8 × 揭示距离 20/30/45 m。
滚动规划按左右优先各运行一次，所有失败保留。这是固定网格的结果，不是总体统计概率或真实车辆验证。</p>
<p>每 100 ms 从当前状态重新筛选候选，反馈控制每 20 ms 更新。预测器与 OCD 不同；预测可行不等于车辆必然安全。
未知表示有限搜索未找到合格机动，不表示物理不可行。</p><p><a href="summary.json">汇总</a> · <a href="scenarios.json">场景与分组</a> · <a href="sensitivity.json">敏感性检查</a> · <a href="overview.png">结果矩阵</a></p>
<table><tr><th>场景</th><th>控制器 / 优先方向</th><th>状态</th><th>避障</th><th>恢复</th><th>规划 P95 ms</th><th>类别</th><th>回放</th></tr>'''
    rows=[]
    for r in results:
        rows.append(f'<tr><td>{r["group_id"]}</td><td>{r["controller"]} / {r["preferred_side"]}</td>'
            f'<td class="{"ok" if r["success"] else "bad"}">{r["status"]}</td><td>{r["avoidance_success"]}</td>'
            f'<td>{r["stable_recovery"]}</td><td>{r.get("planner_p95_ms",0):.2f}</td><td>{categories[r["group_id"]]}</td>'
            f'<td><a href="{r["path"]}/replay.html">播放</a> · <a href="{r["path"]}/metrics.json">指标</a></td></tr>')
    links="".join(f'<li><a href="{r["path"]}/replay.html">{r["variant"]}: {r["status"]}</a></li>' for r in sensitivity)
    (args.output/"index.html").write_text(header+"\n".join(rows)+"</table><h2>独立检查</h2><ul>"+links+"</ul></html>")
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    groups=[s["group_id"] for s in scenario_index]
    labels=[("aeb",1),("joint",1),("rolling",1),("rolling",-1)]
    data=np.array([[next(2 if r["success"] else 1 if r["avoidance_success"] else 0 for r in results
                        if r["group_id"]==g and r["controller"]==mode and r["preferred_side"]==side)
                   for mode,side in labels] for g in groups])
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(figsize=(8,9));im=ax.imshow(data,cmap=ListedColormap(["#c75252","#e0af48","#3a9976"]),vmin=0,vmax=2,aspect="auto")
    ax.set(xticks=range(4),xticklabels=["AEB","Joint fixed","Rolling left","Rolling right"],yticks=range(len(groups)),yticklabels=groups)
    ax.set_title("Finite scenario grid: 0=failure, 1=clearance only, 2=recovered")
    for i in range(len(groups)):
        for j in range(4):ax.text(j,i,str(data[i,j]),ha="center",va="center",color="white")
    fig.tight_layout();fig.savefig(args.output/"overview.png",dpi=160);fig.savefig(args.output/"overview.pdf");plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
