#!/usr/bin/env python3
"""Run the three OCD MF5.2 peak scales with a 15 m/s initial ego speed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_comparison import run

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "ocd_peak_sweep_15mps.json").read_text())
RESULTS = HERE / "results_ocd_peak_sweep_15mps"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, level in enumerate(CONFIG["levels"], 1):
        print(f"[{index}/{len(CONFIG['levels'])}] {level['id']} scale={level['internal_peak_scale']}", flush=True)
        summaries.append(
            run(
                "ocd",
                args.force,
                output_root=RESULTS,
                output_name=level["id"],
                ocd_peak_scale=level["internal_peak_scale"],
                initial_speed_mps=CONFIG["initial_speed_mps"],
            )
        )
    (RESULTS / "batch_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
