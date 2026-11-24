"""Official pretrained inference; synthetic fixtures are ONLY interface checks.

Supports real NPZ input from the official observation adapter/preprocessor.
Batch size is intentionally one to avoid upstream's CFG repeat ordering issue.
"""
import argparse
import hashlib
import json
import platform
import time

from common import ROOT, load_model
import numpy as np
import torch
from flow_planner.data.dataset.nuplan import NuPlanDataSample


def synthetic_scene():
    """A structured straight road with one moving neighbor; not a nuPlan log."""
    data = NuPlanDataSample(batched=True)
    shapes = {
        "ego_past": (1, 21, 14), "ego_current": (1, 16),
        "neighbor_past": (1, 32, 21, 11), "lanes": (1, 70, 20, 12),
        "lanes_speedlimit": (1, 70, 1), "lanes_has_speedlimit": (1, 70, 1),
        "routes": (1, 25, 20, 12), "routes_speedlimit": (1, 25, 1),
        "routes_has_speedlimit": (1, 25, 1), "map_objects": (1, 5, 10),
    }
    for key, shape in shapes.items():
        dtype = torch.bool if "has_speedlimit" in key else torch.float32
        setattr(data, key, torch.zeros(shape, dtype=dtype))
    data.ego_current[0, 2] = 1
    data.ego_current[0, 4] = 8
    data.ego_past[0, :, 0] = torch.linspace(-16, 0, 21)
    data.ego_past[0, :, 2] = 1
    data.ego_past[0, :, 4] = 8
    neighbor = data.neighbor_past[0, 0]
    neighbor[:, 0] = torch.linspace(16, 28, 21)
    neighbor[:, 1] = 3.5
    neighbor[:, 2] = 1
    neighbor[:, 4] = 6
    neighbor[:, 6:8] = torch.tensor([2.0, 4.5])
    neighbor[:, 8] = 1
    for lane_index, lateral in enumerate([-3.5, 0, 3.5]):
        lane = data.lanes[0, lane_index]
        lane[:, 0] = torch.linspace(-20, 100, 20)
        lane[:, 1] = lateral
        lane[:, 2] = 120 / 19
        lane[:, 5] = 1.75
        lane[:, 7] = -1.75
        lane[:, 11] = 1  # unknown traffic light
        data.lanes_speedlimit[0, lane_index] = 13.4
        data.lanes_has_speedlimit[0, lane_index] = True
    data.routes[0, 0] = data.lanes[0, 1]
    data.routes_speedlimit[0, 0] = 13.4
    data.routes_has_speedlimit[0, 0] = True
    return data


def load_npz(path):
    mapping = {
        "ego_agent_past": "ego_past", "ego_current_state": "ego_current",
        "neighbor_agents_past": "neighbor_past", "lanes": "lanes",
        "lanes_speed_limit": "lanes_speedlimit",
        "lanes_has_speed_limit": "lanes_has_speedlimit",
        "route_lanes": "routes", "route_lanes_speed_limit": "routes_speedlimit",
        "route_lanes_has_speed_limit": "routes_has_speedlimit",
        "static_objects": "map_objects",
    }
    with np.load(path, allow_pickle=False) as arrays:
        data = NuPlanDataSample(batched=True)
        batched = arrays["ego_current_state"].ndim == 2
        for source, target in mapping.items():
            value = torch.from_numpy(arrays[source].copy())
            value = value if batched else value.unsqueeze(0)
            if target == "neighbor_past":
                value = value[:, :32]
            value = value.bool() if "has_speedlimit" in target else value.float()
            setattr(data, target, value)
        if data.ego_current.shape[0] != 1:
            raise ValueError("Use one scene per batch; upstream CFG ordering is unsafe for B > 1")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=str)
    parser.add_argument("--output", type=str, default="artifacts/synthetic_smoke")
    parser.add_argument("--seed", type=int, default=3402)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--cfg-weight", type=float, default=1.0)
    parser.add_argument("--no-cfg", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.steps < 1:
        parser.error("runs and steps must be positive, warmup must be nonnegative")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run from a host terminal with GPU access")
    torch.manual_seed(args.seed)
    model, config = load_model(cfg_weight=args.cfg_weight, steps=args.steps)
    data = load_npz(args.npz) if args.npz else synthetic_scene()
    data.to("cuda")
    outdir = ROOT / args.output
    outdir.mkdir(parents=True, exist_ok=True)
    counters = {"encoder": 0, "decoder": 0}
    def hook(name):
        def record(*_):
            counters[name] += 1
        return record
    handles = [model.model_encoder.register_forward_hook(hook("encoder")),
               model.model_decoder.register_forward_hook(hook("decoder"))]
    def infer(seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return model(data, mode="inference", use_cfg=not args.no_cfg,
                     cfg_weight=args.cfg_weight)
    with torch.inference_mode():
        reference = infer(args.seed)
        repeated = infer(args.seed)
        repeat_error = (reference - repeated).abs().max().item()
        assert torch.isfinite(reference).all(), "Non-finite model output"
        assert tuple(reference.shape) == (1, 1, 80, 4), reference.shape
        for _ in range(args.warmup):
            infer()
        torch.cuda.synchronize()
        counters.update(encoder=0, decoder=0)
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = infer()
            torch.cuda.synchronize()
            durations.append(1000 * (time.perf_counter() - start))
            assert torch.isfinite(prediction).all(), "Non-finite model output"
        measured_calls = {key: value / args.runs for key, value in counters.items()}
        peak_allocated = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved = torch.cuda.max_memory_reserved() / 2**20
        samples = torch.cat([infer(args.seed + i).cpu() for i in range(5)], dim=0).numpy()
    for handle in handles:
        handle.remove()
    report = {
        "input_kind": "real_npz" if args.npz else "synthetic_interface_fixture",
        "input_path": args.npz,
        "claim_scope": "Model inference only; not closed-loop performance or paper score reproduction",
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": hashlib.sha256((ROOT / "checkpoints/model.pth").read_bytes()).hexdigest(),
        "weights": "official flat state_dict; module prefix removed; strict=True", "seed": args.seed,
        "cfg_enabled": not args.no_cfg, "cfg_weight": args.cfg_weight,
        "ode_method": "midpoint", "ode_steps": args.steps,
        "calls_per_inference": measured_calls,
        "output_shape": list(reference.shape), "same_seed_max_abs_error": repeat_error,
        "runs": args.runs, "warmup": args.warmup,
        "latency_scope": "GPU-resident structured input to GPU trajectory, includes normalization/encoding/ODE/assembly, synchronized wall time",
        "latency_ms": {"mean": float(np.mean(durations)), "p50": float(np.percentile(durations, 50)),
                       "p95": float(np.percentile(durations, 95)), "p99": float(np.percentile(durations, 99))},
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "raw_latency_ms": durations,
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(outdir / "trajectories.npz", trajectories=samples)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    lanes = data.lanes[0].cpu().numpy()
    for lane in lanes:
        if np.any(lane):
            ax.plot(lane[:, 0], lane[:, 1], color="0.7", linewidth=1)
    for i, sample in enumerate(samples):
        ax.plot(sample[0, :, 0], sample[0, :, 1], label=f"seed {args.seed+i}")
    ax.scatter([0], [0], c="black", label="ego origin")
    ax.set(xlabel="Forward x (m)", ylabel="Left y (m)",
           title="Official weights: synthetic interface check (NOT a nuPlan evaluation)" if not args.npz
           else "Official weights: real structured input (open-loop inference)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "trajectories.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in report.items() if key != "raw_latency_ms"}, indent=2))


if __name__ == "__main__":
    main()
