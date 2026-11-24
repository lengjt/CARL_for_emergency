"""Dependency-light inference for the OCD Python 3.10 environment."""

import json
import math
from pathlib import Path
import numpy as np


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _linear(x, weight, bias):
    return x @ weight.T + bias


def _layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


class NumpyCFM:
    def __init__(self, path):
        archive = np.load(Path(path), allow_pickle=True)
        self.weights = {key.replace("__", "."): archive[key] for key in archive.files
                        if key != "metadata_json"}
        self.metadata = json.loads(str(archive["metadata_json"][0]))
        self.condition_mean = np.asarray(self.metadata["condition_mean"], dtype=np.float64)
        self.condition_std = np.asarray(self.metadata["condition_std"], dtype=np.float64)
        self.target_mean = np.asarray(self.metadata["target_mean"], dtype=np.float64)
        self.target_std = np.asarray(self.metadata["target_std"], dtype=np.float64)
        self.target_clip = np.asarray(self.metadata["target_clip"], dtype=np.float64)
        self.blocks = int(self.metadata["residual_blocks"])

    def velocity(self, x, t, condition):
        t_column = np.full((len(x), 1), float(t))
        time_features = np.concatenate((t_column, np.sin(math.pi * t_column),
                                        np.cos(math.pi * t_column)), axis=1)
        c = np.broadcast_to(condition, (len(x), len(condition)))
        h = _silu(_linear(np.concatenate((x, c, time_features), axis=1),
                          self.weights["input.weight"], self.weights["input.bias"]))
        for index in range(self.blocks):
            prefix = f"blocks.{index}."
            z = _layer_norm(h, self.weights[prefix + "norm.weight"],
                            self.weights[prefix + "norm.bias"])
            z = _silu(_linear(z, self.weights[prefix + "linear1.weight"],
                              self.weights[prefix + "linear1.bias"]))
            z = _linear(z, self.weights[prefix + "linear2.weight"],
                        self.weights[prefix + "linear2.bias"])
            h = h + z / math.sqrt(2.0)
        h = _layer_norm(h, self.weights["output_norm.weight"],
                        self.weights["output_norm.bias"])
        return _linear(h, self.weights["output.weight"], self.weights["output.bias"])

    def sample(self, condition_raw, count, steps, seed):
        condition = (np.asarray(condition_raw, dtype=np.float64) - self.condition_mean) / self.condition_std
        rng = np.random.default_rng(int(seed))
        x = rng.normal(size=(int(count), len(self.target_mean)))
        dt = 1.0 / int(steps)
        for index in range(int(steps)):
            t = (index + 0.5) * dt
            x = x + dt * self.velocity(x, t, condition)
        raw = x * self.target_std + self.target_mean
        return np.clip(raw, self.target_clip[:, 0], self.target_clip[:, 1])
