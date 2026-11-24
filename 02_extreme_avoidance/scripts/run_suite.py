#!/usr/bin/env python3
"""Run the frozen representative cases; preserve successes and failures equally."""
import argparse
from dataclasses import replace
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extreme_avoidance.scenario import Scenario
from extreme_avoidance.experiment import run,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for name in ("straight_popup","straight_critical","straight_low_grip",
                 "curve_grip_drop","curve_obstacle_low_grip","split_grip"):
        modes=("cruise",) if name in ("curve_grip_drop","split_grip") else ("aeb","steer","joint")
        for mode in modes:
            env,metrics,meta=run(Scenario.load(name),mode)
            tag=name+"_"+mode
            save(env,metrics,meta,args.output/tag)
            results.append({"path":tag,"scenario":name,"controller":mode,**metrics})
            (args.output/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
            print(tag,metrics["status"],"avoidance",metrics["avoidance_success"],
                  "recovery",metrics["stable_recovery"],flush=True)
    # Check outcome robustness at half the physics time step in the contrast case.
    convergence=[]
    for mode in ("aeb","steer","joint"):
        sc=replace(Scenario.load("straight_critical"),dt=.0005)
        env,metrics,meta=run(sc,mode)
        tag="straight_critical_"+mode+"_half_step"
        save(env,metrics,meta,args.output/tag,visual=False)
        original=next(r for r in results if r["path"]=="straight_critical_"+mode)
        convergence.append({"controller":mode,"status_1ms":original["status"],"status_05ms":metrics["status"],
                            "same_outcome":original["status"]==metrics["status"],
                            "final_s_difference_m":metrics["final_s_m"]-original["final_s_m"]})
        print(tag,metrics["status"],flush=True)
    (args.output/"step_comparison.json").write_text(json.dumps(convergence,indent=2)+"\n")
    rows=[]
    for r in results:
        clearance="不适用" if r["avoidance_success"] is None else ("是" if r["avoidance_success"] else "否")
        rows.append(f'<tr><td>{html.escape(r["scenario"])}</td><td>{r["controller"]}</td>'
                    f'<td>{r["status"]}</td><td>{clearance}</td><td>{"是" if r["stable_recovery"] else "否"}</td>'
                    f'<td><a href="{r["path"]}/replay.html">动态回放</a> · '
                    f'<a href="{r["path"]}/trajectory.png">曲线</a> · <a href="{r["path"]}/metrics.json">指标</a></td></tr>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>极限避障实验</title>
<style>body{font:16px system-ui;color:#182c45;background:#f5f7fa;margin:32px auto;max-width:1200px;padding:0 20px}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #d6deea;text-align:left}
a{color:#145fc0}p{line-height:1.7}</style><h1>Python + Open Car Dynamics 极限避障实验</h1>
<p>固定合成研究车辆；所有成功与失败均保留。障碍处理完成与稳定恢复独立记录。
这是单组确定性参数的工程实验，不是统计成功率或实车验证。</p>
<p>建议先比较 straight_critical 的三种控制器，再看 straight_low_grip / joint 的恢复失败。
cruise 无障碍场景用于检查弯道附着切换和道路跟踪。</p>
<table><tr><th>场景</th><th>控制器</th><th>终止状态</th><th>曾完成障碍处理</th><th>稳定恢复</th><th>结果</th></tr>'''
    (args.output/"index.html").write_text(page+"\n".join(rows)+"</table></html>")
    if not all(x["same_outcome"] for x in convergence):
        raise SystemExit("Outcome changed on step halving; inspect step_comparison.json")

if __name__=="__main__":main()
