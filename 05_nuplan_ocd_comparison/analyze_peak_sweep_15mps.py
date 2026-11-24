#!/usr/bin/env python3
"""Analyze the 15 m/s OCD MF5.2 peak-scale sweep."""
import json
from pathlib import Path

import analyze_peak_sweep as analysis

HERE = Path(__file__).resolve().parent
analysis.ROOT = HERE / "results_ocd_peak_sweep_15mps"
analysis.CONFIG = json.loads((HERE / "ocd_peak_sweep_15mps.json").read_text())

if __name__ == "__main__":
    analysis.main()
