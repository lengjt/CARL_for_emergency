#!/usr/bin/env python3
"""Run frozen grouped test scenarios through traditional and learned candidate planners."""

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OCD_ROOT = ROOT.parent / "02_extreme_avoidance"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OCD_ROOT))

from cfm_model.io import manifest, read_json, read_jsonl, write_json
from cfm_model.benchmark_index import write_benchmark_index
from cfm_model.planner import make_cfm_planner, make_hybrid_planner, make_retrieval_planner
from cfm_model.schema import CONDITION_FIELDS, TARGET_FIELDS
from extreme_avoidance.environment import AvoidanceEnv
from extreme_avoidance.experiment import run, save
from extreme_avoidance.planner import RollingPlanner
from extreme_avoidance.scenario import Scenario


CFMPlanner = make_cfm_planner(RollingPlanner)
RetrievalPlanner = make_retrieval_planner(RollingPlanner)
HybridPlanner = make_hybrid_planner(RollingPlanner)


def append_jsonl(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def run_custom(scenario, controller):
    env = AvoidanceEnv(scenario)
    env.rows[0].update(planner_replanned=False, planner_fallback=False,
                       planner_selected_offset_m=0.0, planner_latency_ms=0.0)
    latencies = []
    wall_start = time.perf_counter()
    while env.status == "running":
        started = time.perf_counter()
        action = controller(env.observe())
        latencies.append(time.perf_counter() - started)
        env.step(action)
        env.rows[-1].update({"planner_latency_ms": 0.0, **controller.last_event})
    metrics = env.metrics()
    metrics.update(controller.metrics())
    metrics.update({
        "wall_time_s": time.perf_counter() - wall_start,
        "controller_p95_ms": float(np.percentile(latencies, 95) * 1000.0),
        "controller_p99_ms": float(np.percentile(latencies, 99) * 1000.0),
        "controller_deadline_miss_rate": float(np.mean(np.asarray(latencies) > scenario.control_dt)),
        "controller_timing_scope": "observation assembly + controller call; excludes OCD stepping and disk output",
    })
    env.planning_journal = controller.journal
    return env, metrics


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["controller"] if row["controller"] not in ("cfm", "hybrid")
               else f"{row['controller']}_seed_{row['model_seed']}")
        grouped[key].append(row)
    result = {}
    for key, values in grouped.items():
        count = len(values)
        result[key] = {
            "episodes": count,
            "avoidance_successes": sum(bool(row["avoidance_success"]) for row in values),
            "stable_recoveries": sum(row["stable_recovery"] for row in values),
            "valid_successes": sum(row["success"] for row in values),
            "constraint_violations": sum(row["constraint_violation"] for row in values),
            "collisions": sum(row["collision"] for row in values),
            "offroad": sum(row["offroad"] for row in values),
            "recovery_failures": sum(row["recovery_failed"] for row in values),
            "fallback_rate_mean": float(np.mean([row.get("planner_fallback_rate", 0.0) for row in values])),
            "planner_p95_ms_mean": float(np.mean([row.get("planner_p95_ms", 0.0) for row in values])),
            "planner_p99_ms_max": float(max(row.get("planner_p99_ms", 0.0) for row in values)),
            "deadline_miss_rate_max": float(max(row.get("planner_deadline_miss_rate", 0.0) for row in values)),
            "candidate_generation_p95_ms_mean": float(np.mean([
                row.get("candidate_generation_p95_ms", 0.0) for row in values])),
        }
        result[key].update({
            "avoidance_rate": result[key]["avoidance_successes"] / count,
            "stable_recovery_rate": result[key]["stable_recoveries"] / count,
            "valid_success_rate": result[key]["valid_successes"] / count,
            "constraint_violation_rate": result[key]["constraint_violations"] / count,
        })
    cfm_values = [row for row in rows if row["controller"] == "cfm"]
    if cfm_values:
        result["cfm_all_seeds"] = {
            "episodes": len(cfm_values),
            "scenario_groups": len({row["group_id"] for row in cfm_values}),
            "model_seeds": sorted({row["model_seed"] for row in cfm_values}),
            "valid_success_rate": float(np.mean([row["success"] for row in cfm_values])),
            "avoidance_rate": float(np.mean([bool(row["avoidance_success"]) for row in cfm_values])),
            "stable_recovery_rate": float(np.mean([row["stable_recovery"] for row in cfm_values])),
            "constraint_violation_rate": float(np.mean([row["constraint_violation"] for row in cfm_values])),
            "planner_p95_ms_mean": float(np.mean([row["planner_p95_ms"] for row in cfm_values])),
            "planner_p99_ms_max": float(max(row["planner_p99_ms"] for row in cfm_values)),
        }
    hybrid_values = [row for row in rows if row["controller"] == "hybrid"]
    if hybrid_values:
        result["hybrid_all_seeds"] = {
            "episodes": len(hybrid_values),
            "scenario_groups": len({row["group_id"] for row in hybrid_values}),
            "model_seeds": sorted({row["model_seed"] for row in hybrid_values}),
            "valid_success_rate": float(np.mean([row["success"] for row in hybrid_values])),
            "avoidance_rate": float(np.mean([bool(row["avoidance_success"]) for row in hybrid_values])),
            "stable_recovery_rate": float(np.mean([row["stable_recovery"] for row in hybrid_values])),
            "constraint_violation_rate": float(np.mean([row["constraint_violation"] for row in hybrid_values])),
            "planner_p95_ms_mean": float(np.mean([row["planner_p95_ms"] for row in hybrid_values])),
            "planner_p99_ms_max": float(max(row["planner_p99_ms"] for row in hybrid_values)),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.json")
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--model-seed", type=int, action="append")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    model_config = read_json(ROOT / "configs/model.json")
    planner_config = read_json(OCD_ROOT / "configs/planner.json")
    planner_config["deadline_s"] = config["planning_deadline_s"]
    scenario_rows = [row for row in read_json(args.dataset / "scenarios.json") if row["split"] == "test"]
    if args.max_scenarios is not None:
        scenario_rows = scenario_rows[:args.max_scenarios]
    demonstrations = [row for row in read_jsonl(args.dataset / "demonstrations.jsonl")
                      if row["split"] == "train"]
    retrieval_conditions = np.asarray([[row[name] for name in CONDITION_FIELDS]
                                       for row in demonstrations], dtype=np.float64)
    retrieval_targets = np.asarray([[row[name] for name in TARGET_FIELDS]
                                    for row in demonstrations], dtype=np.float64)
    retrieval_mean = retrieval_conditions.mean(axis=0)
    retrieval_std = np.maximum(retrieval_conditions.std(axis=0), 1e-4)
    seeds = args.model_seed or config["cfm_model_seeds"]
    write_json(args.output / "benchmark_config.json", {
        "benchmark": config, "planner": planner_config, "test_scenarios": len(scenario_rows),
        "dataset": str(args.dataset), "training": str(args.training),
        "python": platform.python_version(), "python_executable": sys.executable,
    })
    episodes = []
    episode_path = args.output / "episodes.jsonl"
    for number, item in enumerate(scenario_rows, 1):
        scenario = Scenario.from_dict(item["scenario"])
        specifications = [("aeb", None), ("joint", None), ("rolling", None),
                          ("retrieval", None)] + [("cfm", seed) for seed in seeds] + [
                              ("hybrid", seed) for seed in seeds]
        for controller_name, model_seed in specifications:
            tag = item["group_id"] + "_" + controller_name
            if model_seed is not None:
                tag += f"_seed{model_seed}"
            if controller_name in ("aeb", "joint", "rolling"):
                env, metrics, metadata = run(scenario, controller_name, side=1,
                                             planner_config=planner_config)
            else:
                if controller_name == "retrieval":
                    controller = RetrievalPlanner(
                        scenario.road, AvoidanceEnv(scenario).backend, scenario.speed,
                        scenario.control_dt, side=1, config=planner_config,
                        conditions=retrieval_conditions, targets=retrieval_targets,
                        condition_mean=retrieval_mean, condition_std=retrieval_std,
                        candidate_count=model_config["candidate_count"])
                else:
                    model_path = args.training / f"seed_{model_seed}" / "model_numpy.npz"
                    planner_type = CFMPlanner if controller_name == "cfm" else HybridPlanner
                    controller = planner_type(
                        scenario.road, AvoidanceEnv(scenario).backend, scenario.speed,
                        scenario.control_dt, side=1, config=planner_config,
                        model_path=model_path, model_seed=model_seed,
                        candidate_count=(model_config["candidate_count"] if controller_name == "cfm"
                                         else model_config["candidate_count"] // 2))
                # Controller and environment must share actuator parameters but hold independent
                # state objects; RollingPlanner reads the live observation for every decision.
                env, metrics = run_custom(scenario, controller)
                metadata = {
                    "scenario": scenario.to_dict(), "controller": {"mode": controller_name},
                    "planner": planner_config, "model_seed": model_seed,
                    "timing_scope": config["timing_scope"],
                }
            save(env, metrics, metadata, args.output / tag, visual="replay")
            record = {"path": tag, "controller": controller_name, "model_seed": model_seed,
                      "group_id": item["group_id"], "family_id": item["family_id"],
                      "variant_id": item["variant_id"], **metrics}
            episodes.append(record)
            append_jsonl(episode_path, record)
            print(f"[{number:02d}/{len(scenario_rows)}] {tag}: {metrics['status']} "
                  f"p95={metrics.get('planner_p95_ms', 0.0):.2f} ms", flush=True)
    summary = {
        "scope": config["primary_unit"],
        "test_families": len({row["family_id"] for row in scenario_rows}),
        "test_scenarios": len(scenario_rows),
        "episodes": len(episodes),
        "results": summarize(episodes),
        "limitations": [
            "Fixed finite test set; rates are not population probabilities.",
            "OCD uses a synthetic research vehicle and has no CarSim or real-vehicle calibration.",
            "Planner timing excludes OCD stepping, replay rendering, and file output.",
        ],
    }
    write_json(args.output / "summary.json", summary)
    write_benchmark_index(args.output, episodes)
    write_json(args.output / "manifest.json", manifest(args.output))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
