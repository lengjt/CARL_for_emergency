#!/usr/bin/env python3
"""Measure the existing screening predictor at 25 m/s and stronger combined inputs."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OCD_ROOT = ROOT.parent / "02_extreme_avoidance"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OCD_ROOT))
sys.path.insert(0, str(OCD_ROOT / "scripts"))

from extreme_avoidance._ocd_native import predict_actions
from extreme_avoidance.planner import predictor_parameters
from extreme_avoidance.scenario import Road, Scenario
from validate_physics import simulate
from cfm_model.io import manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = Scenario(name="cfm_predictor_validation", speed=25.0, duration=5.0,
                    base_scale=0.8, friction_mode="internal_mu", obstacles=(),
                    road=Road(width=1000.0, length=1000.0))
    inputs = ((0.0, -5000.0), (0.04, -1200.0), (0.06, -2400.0), (0.08, -3600.0))
    errors = []
    for scale in (0.5, 0.8):
        for steer, force in inputs:
            tag = f"v25_g{scale:g}_d{steer:g}_f{force:g}"
            scenario = replace(base, base_scale=scale)
            truth, backend = simulate(scenario, lambda _: (steer, force), args.output / f"{tag}.csv")
            actions = np.tile([steer, force], (500, 1))
            actions[:4] = 0.0
            predicted = predict_actions([0, 0, 0, 25, 0, 0, 0, 0], actions, 0.01,
                                        predictor_parameters(backend, scale))
            predicted = predicted[2::2]
            for horizon in (1.0, 2.0, 5.0):
                count = int(horizon / 0.02)
                position = np.hypot(predicted[:count, 0] - truth["x_m"][:count],
                                    predicted[:count, 1] - truth["y_m"][:count])
                errors.append({
                    "speed_mps": 25.0, "capability_scale": scale,
                    "steer_request_rad": steer, "force_request_N": force,
                    "horizon_s": horizon,
                    "max_position_error_m": float(np.max(position)),
                    "max_lateral_error_m": float(np.max(np.abs(
                        predicted[:count, 1] - truth["y_m"][:count]))),
                    "max_speed_error_mps": float(np.max(np.abs(
                        predicted[:count, 3] - truth["v_x_mps"][:count]))),
                    "max_yaw_error_rad": float(np.max(np.abs(
                        predicted[:count, 2] - truth["psi_rad"][:count]))),
                })
            np.savez_compressed(args.output / f"{tag}.npz", prediction=predicted,
                                truth=np.stack([truth[name] for name in
                                                ("x_m", "y_m", "psi_rad", "v_x_mps")], axis=1))
    worst = {str(int(horizon)): {
        field: max(row[field] for row in errors if row["horizon_s"] == horizon)
        for field in ("max_position_error_m", "max_lateral_error_m",
                      "max_speed_error_mps", "max_yaw_error_rad")}
        for horizon in (1.0, 2.0, 5.0)}
    report = {
        "scope": "25 m/s, two capability scales, pure braking and three combined steer/brake requests",
        "conditions": 8, "horizons_s": [1.0, 2.0, 5.0], "errors": errors,
        "worst_by_horizon": worst,
        "claim_boundary": "Finite open-loop sweep; measured errors are not confidence or safety bounds.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_json(args.output / "manifest.json", manifest(args.output))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
