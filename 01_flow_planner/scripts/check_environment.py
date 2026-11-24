"""Validate official planner construction/loading, without a dataset or training."""
import importlib.metadata
import json
import platform
import subprocess

from common import ROOT, inference_config
from omegaconf import OmegaConf
import torch
from flow_planner.planner import FlowPlanner
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not visible in this process; run outside the restricted sandbox")
    torch.set_num_threads(4)
    outdir = ROOT / "artifacts"
    outdir.mkdir(exist_ok=True)
    config_path = outdir / "checked_inference_config.yaml"
    OmegaConf.save(inference_config(), config_path)
    with torch.inference_mode():
        planner = FlowPlanner(
            config_path=str(config_path), ckpt_path=str(ROOT / "checkpoints/model.pth"),
            past_trajectory_sampling=TrajectorySampling(num_poses=20, time_horizon=2),
            future_trajectory_sampling=TrajectorySampling(num_poses=80, time_horizon=8),
            enable_ema=False, device="cuda",
        )
        # Initialization only assigns the map handle and loads weights.
        # No scenario is constructed and no map-dependent computation is claimed.
        planner.initialize(PlannerInitialization(
            route_roadblock_ids=[], mission_goal=StateSE2(0, 0, 0), map_api=None))
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "flow-matching", "hydra-core", "numpy", "scipy",
        "pytorch-lightning", "setuptools", "tensorboard")}
    def revision(repo):
        return subprocess.check_output(["git", "-C", str(ROOT / "upstream" / repo),
                                        "rev-parse", "HEAD"], text=True).strip()
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "gpu": torch.cuda.get_device_name(0),
              "gpu_total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
              "cuda_runtime": torch.version.cuda, "versions": versions,
              "flow_planner_revision": revision("Flow-Planner"),
              "nuplan_revision": revision("nuplan-devkit"),
              "weight_revision": "b627934a8db3021892afd13f5278ba0149532efc",
              "official_planner_constructed_and_weights_loaded": True,
              "real_scenario_simulated": False,
              "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()}
    (outdir / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
