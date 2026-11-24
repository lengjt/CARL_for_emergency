"""PyTorch residual MLP vector field and export used only in the training environment."""

import math
import numpy as np
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)

    def forward(self, x):
        h = self.norm(x)
        h = torch.nn.functional.silu(self.linear1(h))
        return x + self.linear2(h) / math.sqrt(2.0)


class VectorField(nn.Module):
    def __init__(self, condition_dim, target_dim, width=128, blocks=3):
        super().__init__()
        self.condition_dim = condition_dim
        self.target_dim = target_dim
        self.input = nn.Linear(condition_dim + target_dim + 3, width)
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(blocks)])
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, target_dim)

    def forward(self, x, t, condition):
        if t.ndim == 1:
            t = t[:, None]
        time_features = torch.cat((t, torch.sin(math.pi * t), torch.cos(math.pi * t)), dim=-1)
        h = torch.nn.functional.silu(self.input(torch.cat((x, condition, time_features), dim=-1)))
        for block in self.blocks:
            h = block(h)
        return self.output(self.output_norm(h))


def export_numpy(model, path, metadata):
    arrays = {key.replace(".", "__"): value.detach().cpu().numpy()
              for key, value in model.state_dict().items()}
    arrays["metadata_json"] = np.asarray([metadata], dtype=object)
    np.savez(path, **arrays)
