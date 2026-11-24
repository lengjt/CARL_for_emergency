"""Compare the primary post-MF52 scale with internal LMUX/LMUY scaling."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from extreme_avoidance.backend import OCDBackend, PREFIX
from extreme_avoidance.scenario import Scenario
from extreme_avoidance._ocd_native import TireBench


def curve_metrics(tire, fz, external_scale):
    kappas = np.linspace(-0.3, 0.3, 301)
    alphas = np.linspace(-0.25, 0.25, 301)
    fx = np.array([tire.forces(float(k), 0.0, fz, external_scale)[0] for k in kappas])
    fy = np.array([tire.forces(0.0, float(a), fz, external_scale)[1] for a in alphas])
    ix, iy = int(np.argmax(abs(fx))), int(np.argmax(abs(fy)))
    dk = kappas[151] - kappas[149]
    da = alphas[151] - alphas[149]
    return {
        "peak_abs_Fx_over_Fz": float(abs(fx[ix]) / fz),
        "peak_abs_Fy_over_Fz": float(abs(fy[iy]) / fz),
        "abs_kappa_at_sampled_Fx_peak": float(abs(kappas[ix])),
        "abs_alpha_at_sampled_Fy_peak_rad": float(abs(alphas[iy])),
        "small_slip_Kx_N": float((fx[151] - fx[149]) / dk),
        "small_slip_Ky_N_per_rad": float((fy[151] - fy[149]) / da),
        "combined_at_kappa_0p1_alpha_0p1_N": list(tire.forces(0.1, 0.1, fz, external_scale)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output file already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    backend = OCDBackend(Scenario())
    prefix = PREFIX + "z.tire_model.front_left."
    params = {k[len(prefix):]: v for k, v in backend.params.items() if k.startswith(prefix)}
    definition = json.loads((ROOT / "configs/friction_definition.json").read_text())
    fz = definition["reference_reporting"]["reference_load_N"]
    rows = []
    for scale in (1.0, 0.8, 0.6, 0.5):
        post = curve_metrics(TireBench(params), fz, scale)
        internal_params = dict(params)
        internal_params["LMUX"] = params["LMUX"] * scale
        internal_params["LMUY"] = params["LMUY"] * scale
        internal = curve_metrics(TireBench(internal_params), fz, 1.0)
        rows.append({"scale": scale, "post_mf52_force_scale": post,
                     "internal_LMUX_LMUY_scale": internal})
    payload = {
        "definition": definition,
        "scope": "finite-grid comparison for one frozen tire at the reference load; not road identification",
        "comparison": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
