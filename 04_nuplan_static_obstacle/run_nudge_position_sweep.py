#!/usr/bin/env python3
"""Run a controlled stopped-vehicle position sweep on one official nudge scene."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_official_interplan import build_scenario, run_one, stopped_vehicle_poses

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "nudge_position_sweep.json"
RESULTS = HERE / "results_nudge_position_sweep"


def shifted_pose(pose, longitudinal_m: float, lateral_m: float):
    c, s = math.cos(pose.heading), math.sin(pose.heading)
    return [
        pose.x + longitudinal_m * c - lateral_m * s,
        pose.y + longitudinal_m * s + lateral_m * c,
        pose.heading,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    token = config["source_modified_token"]
    scenario, _ = build_scenario(token)
    original = stopped_vehicle_poses(scenario)[0]
    summaries = []
    for index, variant in enumerate(config["variants"], 1):
        pose = shifted_pose(original, variant["longitudinal_m"], variant["lateral_m"])
        print(f"[{index}/{len(config['variants'])}] {variant['id']} pose={pose}", flush=True)
        summaries.append(
            run_one(
                token,
                force=args.force,
                output_root=RESULTS,
                output_name=variant["id"],
                stopped_vehicle_override=[pose],
                summary_metadata={
                    "sweep_longitudinal_m": variant["longitudinal_m"],
                    "sweep_lateral_m": variant["lateral_m"],
                    "original_stopped_vehicle_pose": [original.x, original.y, original.heading],
                },
            )
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "batch_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
