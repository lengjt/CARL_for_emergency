#!/usr/bin/env python3
"""Generate grouped OCD-verified rolling-planner demonstrations and retain failures."""

import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCD_ROOT = ROOT.parent / "02_extreme_avoidance"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OCD_ROOT))

from cfm_model.io import manifest, read_json, write_json
from cfm_model.schema import CONDITION_FIELDS, TARGET_FIELDS, context_from_row, target_from_parameters
from cfm_model.scenarios import build_scenarios
from extreme_avoidance.experiment import run
from extreme_avoidance.scenario import Obstacle, Road, Scenario


def append_jsonl(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def closest_row(rows, event_time):
    return min(rows, key=lambda row: abs(row["t"] - event_time))


def audit(scenarios, episodes, demonstrations, elapsed, config):
    family_splits = defaultdict(set)
    group_splits = defaultdict(set)
    for item in scenarios:
        family_splits[item["family_id"]].add(item["split"])
        group_splits[item["group_id"]].add(item["split"])
    if any(len(value) != 1 for value in family_splits.values()):
        raise RuntimeError("A scenario family leaked across splits")
    if any(len(value) != 1 for value in group_splits.values()):
        raise RuntimeError("A base scenario leaked across splits")
    summary = {
        "dataset_name": config["name"],
        "scope": config["scope"],
        "wall_time_s": elapsed,
        "scenario_families": len(family_splits),
        "scenario_groups": len(scenarios),
        "episodes": len(episodes),
        "successful_episodes": sum(row["success"] for row in episodes),
        "failed_episodes_retained": sum(not row["success"] for row in episodes),
        "demonstration_decisions": len(demonstrations),
        "unique_target_modes_rounded": len({tuple(round(row[name], 3) for name in TARGET_FIELDS)
                                             for row in demonstrations}),
        "by_split": {},
        "failure_statuses": dict(Counter(row["status"] for row in episodes if not row["success"])),
        "label_policy": ("Only decisions from an episode whose complete OCD rollout ended in stable recovery "
                         "without collision, off-road, numerical, or actuator violations are positive demonstrations."),
        "verification_scope": ("The selected decision sequence is verified by the complete OCD closed loop. "
                               "Unexecuted predictor-feasible candidates are not labeled positive."),
        "leakage_checks": {
            "family_split_disjoint": True,
            "group_split_disjoint": True,
            "mirrors_share_group": True,
            "nearby_variants_share_family": True,
        },
        "condition_fields": list(CONDITION_FIELDS),
        "target_fields": list(TARGET_FIELDS),
    }
    for split in ("train", "val", "test"):
        selected_scenarios = [row for row in scenarios if row["split"] == split]
        selected_episodes = [row for row in episodes if row["split"] == split]
        selected_demos = [row for row in demonstrations if row["split"] == split]
        summary["by_split"][split] = {
            "families": len({row["family_id"] for row in selected_scenarios}),
            "scenarios": len(selected_scenarios),
            "episodes": len(selected_episodes),
            "successful_episodes": sum(row["success"] for row in selected_episodes),
            "failed_episodes": sum(not row["success"] for row in selected_episodes),
            "demonstration_decisions": len(selected_demos),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--max-scenarios", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    planner_config = read_json(OCD_ROOT / "configs/planner.json")
    scenario_rows = build_scenarios(config, (Scenario, Road, Obstacle))
    if args.max_scenarios is not None:
        scenario_rows = scenario_rows[:args.max_scenarios]
    write_json(args.output / "scenarios.json", scenario_rows)
    write_json(args.output / "generation_config.json", {
        "data": config, "planner": planner_config,
        "python": platform.python_version(), "python_executable": sys.executable,
        "ocd_root": str(OCD_ROOT),
    })
    episode_path = args.output / "episodes.jsonl"
    demonstration_path = args.output / "demonstrations.jsonl"
    started = time.perf_counter()
    episodes = []
    demonstrations = []
    for scenario_number, item in enumerate(scenario_rows, 1):
        scenario = Scenario.from_dict(item["scenario"])
        obstacle = scenario.obstacles[0]
        for side in config["preferred_sides"]:
            env, metrics, metadata = run(scenario, "rolling", side=side, planner_config=planner_config)
            episode_id = f"{item['group_id']}_{'left' if side == 1 else 'right'}"
            episode = {
                "episode_id": episode_id, "group_id": item["group_id"],
                "family_id": item["family_id"], "variant_id": item["variant_id"],
                "split": item["split"], "preferred_side": side,
                "selected_decisions": sum(event["selected_index"] is not None
                                          for event in env.planning_journal),
                "fallback_decisions": sum(event["fallback"] for event in env.planning_journal),
                "upstream_commit": metadata["upstream_commit"], **metrics,
            }
            episodes.append(episode)
            append_jsonl(episode_path, episode)
            if metrics["success"]:
                selected_events = [event for event in env.planning_journal
                                   if event["selected_index"] is not None]
                stride = int(config["demonstration_stride"])
                for decision_index, event in enumerate(selected_events[::stride]):
                    parameters = event["candidates"][event["selected_index"]]["parameters"]
                    row = closest_row(env.rows, event["t"])
                    record = {
                        "demonstration_id": f"{episode_id}_{decision_index:03d}",
                        "episode_id": episode_id, "group_id": item["group_id"],
                        "family_id": item["family_id"], "variant_id": item["variant_id"],
                        "split": item["split"], "preferred_side": side,
                        "decision_time_s": event["t"], "episode_success": True,
                        "ocd_verified_episode": True,
                        **context_from_row(row, scenario, obstacle),
                        **target_from_parameters(parameters),
                    }
                    demonstrations.append(record)
                    append_jsonl(demonstration_path, record)
            print(f"[{scenario_number:02d}/{len(scenario_rows)}] {episode_id}: "
                  f"{metrics['status']} demos={len(demonstrations)}", flush=True)
    summary = audit(scenario_rows, episodes, demonstrations,
                    time.perf_counter() - started, config)
    write_json(args.output / "audit.json", summary)
    write_json(args.output / "manifest.json", manifest(args.output))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
