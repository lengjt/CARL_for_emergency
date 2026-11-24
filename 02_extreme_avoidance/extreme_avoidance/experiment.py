"""Run provenance, control-period traces, plots, and a standalone browser replay."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
from .environment import AvoidanceEnv
from .controllers import Baseline
from .scenario import ROOT

def run(scenario, mode="joint", side=1, offset=2.8, brake=3000., ramp_length=None, vehicle_file=None,
        planner_config=None):
    env = AvoidanceEnv(scenario, vehicle_file)
    if mode=="rolling":
        from .planner import RollingPlanner
        control=RollingPlanner(scenario.road,env.backend,scenario.speed,scenario.control_dt,side,planner_config)
        env.rows[0].update(planner_replanned=False,planner_fallback=False,planner_selected_offset_m=0.,planner_latency_ms=0.)
    else:
        control = Baseline(mode, scenario.road, env.backend, scenario.speed, side, offset, brake, ramp_length)
    latencies = []
    start = time.perf_counter()
    while env.status == "running":
        t = time.perf_counter()
        action = control(env.observe())
        latencies.append(time.perf_counter()-t)
        env.step(action)
        if mode=="rolling":
            env.rows[-1].update({"planner_latency_ms":0.,**control.last_event})
    metrics = env.metrics()
    if mode=="rolling":
        metrics.update(control.metrics())
        env.planning_journal=control.journal
    metrics.update({"wall_time_s":time.perf_counter()-start,
                    "controller_p95_ms":float(np.percentile(latencies,95)*1000),
                    "controller_p99_ms":float(np.percentile(latencies,99)*1000),
                    "controller_deadline_miss_rate":float(np.mean(np.array(latencies)>scenario.control_dt)),
                    "controller_timing_scope":"observation assembly + complete controller call (including planner diagnostics); excludes environment stepping"})
    friction_definition = json.loads((ROOT/"configs/friction_definition.json").read_text())
    metadata = {"scenario":scenario.to_dict(),"vehicle":env.backend.config,
                "initial_parameters":env.backend.initial_parameters,
                "controller":{"mode":mode,"side":side,"offset":offset,"brake":brake,"ramp_length":ramp_length},
                "upstream_commit":env.backend.commit,"python":platform.python_version(),
                "python_executable":sys.executable,"python_prefix":sys.prefix,
                "numpy":np.__version__,"platform":platform.platform(),
                "friction_definition":friction_definition,
                "active_friction_mode":scenario.friction_mode,
                "observation_semantics":"structured state; obstacle reveal mask; scale estimate is oracle with optional bias/delay",
                "trace_semantics":"control-period samples; geometry checked every physics step",
                "physics_semantics":"integrated pose/state; wheel-force diagnostics from final RK stage"}
    source_paths = sorted((ROOT/"extreme_avoidance").glob("*.py")) + sorted((ROOT/"native").glob("*"))
    source_paths += sorted((ROOT/"extreme_avoidance").glob("*.html"))
    source_paths += sorted((ROOT/"extreme_avoidance").glob("*.so"))
    metadata["implementation_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in source_paths if p.is_file()}
    metadata["upstream_manifest_sha256"] = hashlib.sha256((ROOT/"upstream/manifest.json").read_bytes()).hexdigest()
    metadata["friction_definition_sha256"] = hashlib.sha256((ROOT/"configs/friction_definition.json").read_bytes()).hexdigest()
    if mode=="rolling":metadata["planner"]=control.config
    return env, metrics, metadata

def save(env, metrics, metadata, directory, visual=True):
    directory = Path(directory)
    # Never overwrite a previous run silently.
    directory.mkdir(parents=True,exist_ok=False)
    with (directory/"trace.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=env.rows[0].keys())
        writer.writeheader();writer.writerows(env.rows)
    for name,data in (("metrics.json",metrics),("config.json",metadata)):
        (directory/name).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    if hasattr(env,"planning_journal"):
        with (directory/"plans.jsonl").open("w") as f:
            for event in env.planning_journal:f.write(json.dumps(event,allow_nan=False)+"\n")
    if visual:
        if visual != "replay":plot(env,metrics,directory/"trajectory.png")
        replay(env,metrics,directory/"replay.html")
    paths=[p for p in directory.iterdir() if p.is_file()]
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

def plot(env,metrics,path):
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    a={k:np.array([r[k] for r in env.rows]) for k in env.rows[0]}
    fig,axes=plt.subplots(4,2,figsize=(13,13))
    ax=axes[0,0]
    sc=env.scenario
    for d in (-sc.road.width/2,0,sc.road.width/2):
        xy=np.array([sc.road.pose(s,d)[:2] for s in np.linspace(-5,sc.road.length,300)])
        ax.plot(xy[:,0],xy[:,1],"--" if d==0 else "-",color="0.65",lw=1)
    for p in env.obstacle_polygons: ax.add_patch(Polygon(p,color="firebrick"))
    ax.plot(a["x_m"],a["y_m"],color="royalblue")
    ax.set_aspect("equal",adjustable="datalim");ax.set(xlabel="x [m]",ylabel="y [m]",title=metrics["status"])
    axes[0,1].plot(a["t"],a["v_x_mps"]);axes[0,1].set(ylabel="vx [m/s]")
    axes[1,0].plot(a["t"],a["steer_request_rad"],label="request")
    axes[1,0].plot(a["t"],a["steering_rad"],label="actual");axes[1,0].set(ylabel="steering [rad]")
    axes[1,1].plot(a["t"],a["force_request_N"],label="request")
    axes[1,1].plot(a["t"],a["force_executed_N"],label="to OCD")
    axes[1,1].plot(a["t"],sum(a["fx_N_"+w] for w in ("fl","fr","rl","rr")),label="sum tire Fx")
    axes[1,1].set(ylabel="force [N]")
    axes[2,0].plot(a["t"],a["beta_rad"],label="beta [rad]")
    axes[2,0].plot(a["t"],a["psi_dot_radps"],label="yaw rate [rad/s]")
    for w in ("fl","fr","rl","rr"): axes[2,1].plot(a["t"],a["scale_true_"+w],label=w)
    axes[2,1].set(ylabel="capability scale ("+sc.friction_mode+")")
    for w in ("fl","fr","rl","rr"):
        axes[3,0].plot(a["t"][1:],a["kappa_"+w][1:],label=w)
        axes[3,1].plot(a["t"][1:],a["fz_N_"+w][1:],label=w)
    axes[3,0].set(ylabel="longitudinal slip")
    axes[3,1].set(ylabel="vertical tire load [N]")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax!=axes[0,0]: ax.set_xlabel("time [s]")
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    fig.suptitle(f"{sc.name} | OCD FX / PT1 / double track / MF52")
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

def replay(env,metrics,path):
    sc=env.scenario
    visible=set()
    rows=[]
    for r in env.rows:
        visible.update(i for i,o in enumerate(sc.obstacles) if o.s-r["s"]<=o.reveal_distance)
        rows.append({**r,"visible_obstacle_indices":sorted(visible)})
    payload={"rows":rows,"obstacles":[p.tolist() for p in env.obstacle_polygons],
             "road":[[sc.road.pose(s,d)[:2] for s in np.linspace(-5,sc.road.length,300)]
                     for d in (-sc.road.width/2,0,sc.road.width/2)],
             "length":env.backend.length,"width":env.backend.width,"name":sc.name,"status":metrics["status"],
             "metrics":metrics,"base_scale":sc.base_scale,"friction_mode":sc.friction_mode,"patches":[]}
    for p in sc.patches:
        lo,hi=max(-sc.road.width/2,p.d_min),min(sc.road.width/2,p.d_max)
        ss=np.linspace(max(-5,p.s_start),min(sc.road.length,p.s_end),100)
        polygon=[sc.road.pose(s,lo)[:2] for s in ss]+[sc.road.pose(s,hi)[:2] for s in ss[::-1]]
        payload["patches"].append({"polygon":polygon,"scale":p.scale,"start":p.t_start,"end":p.t_end})
    template=(ROOT/"extreme_avoidance/replay_template.html").read_text()
    path.write_text(template.replace("__PAYLOAD__",json.dumps(payload,allow_nan=False).replace("<","\\u003c")))
