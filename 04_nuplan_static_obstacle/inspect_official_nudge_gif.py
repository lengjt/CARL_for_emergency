#!/usr/bin/env python3
"""Create a contact sheet of the official Flow-Planner nudge GIF."""
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/flow-planner-mpl")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = Path("/home/ljt23/research_ca/01_flow_planner/upstream/Flow-Planner/assets/gif/interplan/nudge.gif")


def main():
    image = Image.open(SOURCE)
    indices = [0, image.n_frames // 4, image.n_frames // 2, 3 * image.n_frames // 4, image.n_frames - 1]
    fig, axes = plt.subplots(1, len(indices), figsize=(20, 4), dpi=120)
    for ax, index in zip(axes, indices):
        image.seek(index)
        ax.imshow(image.convert("RGBA"))
        ax.set_title(f"frame {index} / t={index * 0.05:.2f}s")
        ax.axis("off")
    fig.tight_layout()
    output = HERE / "official_nudge_contact_sheet.png"
    fig.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
