#!/usr/bin/env python3
"""Train small conditional flow-matching vector fields and export NumPy inference weights."""

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfm_model.io import manifest, read_json, read_jsonl, write_json
from cfm_model.network import VectorField, export_numpy
from cfm_model.schema import CONDITION_FIELDS, TARGET_CLIP, TARGET_FIELDS


def array(rows, fields):
    return np.asarray([[row[name] for name in fields] for row in rows], dtype=np.float32)


def normalized_data(rows, split, c_mean, c_std, y_mean, y_std):
    selected = [row for row in rows if row["split"] == split]
    c = (array(selected, CONDITION_FIELDS) - c_mean) / c_std
    y = (array(selected, TARGET_FIELDS) - y_mean) / y_std
    return torch.from_numpy(c), torch.from_numpy(y)


@torch.no_grad()
def validation_loss(model, condition, target, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed + 900001)
    total = 0.0
    count = 0
    batch = 1024
    for start in range(0, len(target), batch):
        c = condition[start:start + batch].to(device)
        x1 = target[start:start + batch].to(device)
        x0 = torch.randn(x1.shape, generator=generator, device=device)
        t = torch.rand((len(x1), 1), generator=generator, device=device)
        xt = (1.0 - t) * x0 + t * x1
        loss = torch.mean((model(xt, t, c) - (x1 - x0)) ** 2)
        total += float(loss) * len(x1)
        count += len(x1)
    return total / max(count, 1)


@torch.no_grad()
def best_of_k_error(model, condition, target, target_std, seed, device, candidates=32, steps=8):
    """Report generative set error; it is diagnostic and not a safety or dynamics metric."""
    generator = torch.Generator(device=device).manual_seed(seed + 700003)
    normalized_best = []
    raw_absolute = []
    batch = 128
    scale = torch.as_tensor(target_std, device=device)
    for start in range(0, len(target), batch):
        c = condition[start:start + batch].to(device)
        truth = target[start:start + batch].to(device)
        size = len(truth)
        x = torch.randn((size, candidates, truth.shape[1]), generator=generator, device=device)
        expanded_condition = c[:, None, :].expand(-1, candidates, -1).reshape(size * candidates, -1)
        dt = 1.0 / steps
        for index in range(steps):
            t = torch.full((size * candidates, 1), (index + 0.5) * dt, device=device)
            velocity = model(x.reshape(size * candidates, -1), t, expanded_condition)
            x = x + dt * velocity.reshape_as(x)
        difference = x - truth[:, None, :]
        distance = torch.sum(difference * difference, dim=-1)
        best = torch.argmin(distance, dim=1)
        chosen = x[torch.arange(size, device=device), best]
        normalized_best.extend(torch.sqrt(distance[torch.arange(size, device=device), best]).cpu().tolist())
        raw_absolute.append((torch.abs(chosen - truth) * scale).cpu().numpy())
    errors = np.concatenate(raw_absolute, axis=0)
    return {
        "best_of_32_normalized_l2_mean": float(np.mean(normalized_best)),
        "best_of_32_raw_mae": {name: float(errors[:, index].mean())
                                for index, name in enumerate(TARGET_FIELDS)},
        "interpretation": "Nearest generated candidate to each held-out label; not a dynamics or safety bound.",
    }


def train_seed(seed, rows, config, output, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_rows = [row for row in rows if row["split"] == "train"]
    condition_raw = array(train_rows, CONDITION_FIELDS)
    target_raw = array(train_rows, TARGET_FIELDS)
    condition_mean = condition_raw.mean(axis=0)
    condition_std = np.maximum(condition_raw.std(axis=0), 1e-4)
    target_mean = target_raw.mean(axis=0)
    target_std = np.maximum(target_raw.std(axis=0), 1e-4)
    train_c, train_y = normalized_data(rows, "train", condition_mean, condition_std,
                                       target_mean, target_std)
    val_c, val_y = normalized_data(rows, "val", condition_mean, condition_std,
                                   target_mean, target_std)
    model = VectorField(len(CONDITION_FIELDS), len(TARGET_FIELDS), config["width"],
                        config["residual_blocks"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 101)
    device_generator = torch.Generator(device=device).manual_seed(seed + 202)
    history = []
    best = math.inf
    best_step = None
    best_state = None
    stale = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, config["training_steps"] + 1):
        indices = torch.randint(len(train_y), (config["batch_size"],), generator=generator)
        condition = train_c[indices].to(device)
        x1 = train_y[indices].to(device)
        x0 = torch.randn(x1.shape, generator=device_generator, device=device)
        t = torch.rand((len(x1), 1), generator=device_generator, device=device)
        xt = (1.0 - t) * x0 + t * x1
        prediction = model(xt, t, condition)
        loss = torch.mean((prediction - (x1 - x0)) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % config["validation_interval"] == 0:
            model.eval()
            # Reuse the same validation noise/time draws at every checkpoint so that
            # early stopping compares model changes rather than Monte Carlo variation.
            val_loss = validation_loss(model, val_c, val_y, seed, device)
            model.train()
            point = {"step": step, "train_cfm_mse": float(loss), "val_cfm_mse": val_loss}
            history.append(point)
            print(f"seed={seed} step={step} train={float(loss):.6f} val={val_loss:.6f}", flush=True)
            if val_loss < best - 1e-5:
                best = val_loss
                best_step = step
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= config["patience_intervals"]:
                break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_set_error = best_of_k_error(model, val_c, val_y, target_std, seed, device,
                                            candidates=32, steps=config["sampling_steps"])
    test_c, test_y = normalized_data(rows, "test", condition_mean, condition_std,
                                     target_mean, target_std)
    test_set_error = best_of_k_error(model, test_c, test_y, target_std, seed + 1, device,
                                     candidates=32, steps=config["sampling_steps"])
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir()
    torch.save({"model": best_state, "config": config, "seed": seed}, seed_dir / "checkpoint.pt")
    metadata = {
        "name": config["name"], "seed": seed,
        "condition_fields": list(CONDITION_FIELDS), "target_fields": list(TARGET_FIELDS),
        "condition_mean": condition_mean.tolist(), "condition_std": condition_std.tolist(),
        "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
        "target_clip": [list(value) for value in TARGET_CLIP],
        "width": config["width"], "residual_blocks": config["residual_blocks"],
        "candidate_count": config["candidate_count"], "sampling_steps": config["sampling_steps"],
    }
    export_numpy(model, seed_dir / "model_numpy.npz", json.dumps(metadata, separators=(",", ":")))
    result = {
        "seed": seed, "steps_completed": history[-1]["step"], "best_val_cfm_mse": best,
        "best_step": best_step,
        "final_train_cfm_mse": history[-1]["train_cfm_mse"],
        "wall_time_s": time.perf_counter() - started,
        "device": str(device),
        "peak_gpu_memory_MiB": (torch.cuda.max_memory_allocated(device) / 2 ** 20
                                if device.type == "cuda" else 0.0),
        "history": history,
        "validation_set_generation_error": validation_set_error,
        "test_set_generation_error": test_set_error,
    }
    write_json(seed_dir / "training_log.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model.json")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    if args.steps is not None:
        config["training_steps"] = args.steps
    seeds = args.seed or config["seeds"]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    rows = read_jsonl(args.dataset / "demonstrations.jsonl")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    if min(counts.values()) == 0:
        raise RuntimeError(f"Every split needs demonstrations, got {counts}")
    write_json(args.output / "training_config.json", {
        "model": config, "dataset": str(args.dataset), "demonstrations_by_split": counts,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    })
    results = [train_seed(seed, rows, config, args.output, device) for seed in seeds]
    write_json(args.output / "summary.json", {
        "models": len(results), "seeds": seeds, "device": str(device),
        "best_val_cfm_mse_mean": float(np.mean([row["best_val_cfm_mse"] for row in results])),
        "training_wall_time_s_total": sum(row["wall_time_s"] for row in results),
        "runs": results,
    })
    write_json(args.output / "manifest.json", manifest(args.output))


if __name__ == "__main__":
    main()
