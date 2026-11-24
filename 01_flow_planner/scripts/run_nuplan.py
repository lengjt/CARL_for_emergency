"""Sequential official nuPlan closed-loop simulation, without retraining."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from common import ROOT, inference_config
from omegaconf import OmegaConf
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--maps-root", type=Path)
    parser.add_argument("--scenarios", type=int, default=3)
    parser.add_argument("--tokens", nargs="+")
    parser.add_argument("--reactive", action="store_true")
    parser.add_argument("--cfg-weight", type=float, default=1.0)
    parser.add_argument("--compose-only", action="store_true", help="Validate config without reading data")
    args = parser.parse_args()
    if args.scenarios < 1:
        parser.error("scenarios must be positive")
    data_root = args.data_root.resolve()
    maps_root = args.maps_root.resolve() if args.maps_root else data_root / "maps"
    dbs = sorted((data_root / "nuplan-v1.1/splits/mini").glob("*.db"))
    if not args.compose_only:
        if not dbs or not (maps_root / "nuplan-maps-v1.0.json").is_file():
            parser.error("Missing nuPlan mini databases/maps. See docs/DATA_SETUP.md; no simulation was run.")
        if not torch.cuda.is_available():
            parser.error("CUDA unavailable; use a GPU-enabled host terminal")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    outdir = ROOT / "artifacts" / "nuplan"
    outdir.mkdir(parents=True, exist_ok=True)
    config = inference_config()
    config.model.cfg_weight = args.cfg_weight
    config_path = outdir / "inference_config.yaml"
    OmegaConf.save(config, config_path)
    os.environ.update({
        "NUPLAN_DATA_ROOT": str(data_root), "NUPLAN_MAPS_ROOT": str(maps_root),
        "NUPLAN_EXP_ROOT": str(outdir), "HYDRA_FULL_ERROR": "1",
    })
    challenge = "closed_loop_reactive_agents" if args.reactive else "closed_loop_nonreactive_agents"
    overrides = [
        f"+simulation={challenge}", "planner=flow_planner",
        f"planner.flow_planner.config_path={config_path}",
        f"planner.flow_planner.ckpt_path={ROOT / 'checkpoints/model.pth'}",
        "+planner.flow_planner.enable_ema=false",
        "scenario_builder=nuplan_mini", "scenario_filter=all_scenarios",
        f"scenario_filter.limit_total_scenarios={args.scenarios}",
        "scenario_filter.shuffle=false", "scenario_filter.remove_invalid_goals=true",
        "scenario_builder.max_workers=2", "worker=sequential",
        # Sequential worker requires one scheduled CPU task; PyTorch's internal
        # thread count is independently set to four above.
        "number_of_gpus_allocated_per_simulation=1.0", "number_of_cpus_allocated_per_simulation=1",
        "exit_on_failure=true", "verbose=true", "enable_simulation_progress_bar=true",
        "experiment_uid=flow_planner_mini/${now:%Y-%m-%d_%H-%M-%S}",
        "hydra.searchpath=[pkg://flow_planner.nuplan_simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]
    if args.tokens:
        overrides.append("scenario_filter.scenario_tokens=" + json.dumps(args.tokens))
    if args.compose_only:
        overrides += ["--cfg", "job"]
    manifest = {"data_root": str(data_root), "maps_root": str(maps_root),
                "database_files": [str(p) for p in dbs], "overrides": overrides,
                "compose_only": args.compose_only, "cfg_weight": args.cfg_weight}
    (outdir / "last_launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sys.argv = ["run_simulation.py", *overrides]
    runpy.run_module("nuplan.planning.script.run_simulation", run_name="__main__")


if __name__ == "__main__":
    main()
