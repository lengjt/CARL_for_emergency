"""Load the unchanged official model with its released configuration and weights."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "Flow-Planner"))
sys.path.insert(0, str(ROOT / "upstream" / "nuplan-devkit"))

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def inference_config(device="cuda"):
    config = OmegaConf.load(ROOT / "checkpoints" / "model_config.yaml")
    # The released YAML references these two fields but omits the data section.
    # Values come from upstream script/data/dataset/nuplan_data.yaml.
    config = OmegaConf.merge(config, {
        "device": device,
        "data": {"dataset": {"train": {
            "future_downsampling_method": "uniform",
            "predicted_neighbor_num": 0,
        }}},
    })
    return config


def load_model(device="cuda", cfg_weight=None, steps=None):
    config = inference_config(device)
    if cfg_weight is not None:
        # Upstream's forward cfg_weight argument does not override FlowODE's
        # stored weight; set the configuration before instantiation instead.
        config.model.cfg_weight = cfg_weight
    if steps is not None:
        config.model.flow_ode.sample_steps = steps
    model = instantiate(config.model)
    checkpoint = torch.load(ROOT / "checkpoints" / "model.pth",
                            map_location="cpu", weights_only=True)
    if "ema_state_dict" in checkpoint:
        weights = checkpoint["ema_state_dict"]
    elif "model" in checkpoint:
        weights = checkpoint["model"]
    else:
        # The actual released model.pth is a flat OrderedDict (338 tensors),
        # unlike the training checkpoint format assumed by planner.py.
        weights = checkpoint
    weights = {key.removeprefix("module."): value for key, value in weights.items()}
    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, config
