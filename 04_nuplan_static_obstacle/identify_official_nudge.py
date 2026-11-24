#!/usr/bin/env python3
"""Render the ten official interPlan nudge candidates around their parked car."""
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

HERE = Path(__file__).resolve().parent
FLOW_ROOT = Path("/home/ljt23/research_ca/01_flow_planner")
sys.path.insert(0, str(FLOW_ROOT / "upstream/nuplan-devkit"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache/matplotlib"))

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db


CANDIDATES = [
    ("81f400b414215325-s0", 365106.20, 143350.05, -1.95),
    ("948201ecfe2c560a-s0", 365258.93, 143656.40, -2.91),
    ("81f400b414215325-s1", 365097.18, 143327.38, -1.95),
    ("71f182558ee95100-s1", 365418.30, 143895.61, 1.22173),
    ("71f182558ee95100-s2", 365421.00, 143905.34, 1.309),
    ("948201ecfe2c560a-s2", 365285.31, 143662.50, -2.91),
    ("2d62c3139aa95007-s3", 365390.44, 143806.36, 1.274),
    ("2d62c3139aa95007-s4", 365396.40, 143825.40, 1.274),
    ("2d62c3139aa95007-s5", 365403.84, 143848.84, 1.274),
    ("71f182558ee95100-s3", 365415.95, 143965.45, 1.95477),
]


def transform(coords, origin, heading):
    # Rotate so the parked vehicle heading points to the right, matching the GIF.
    points = np.asarray(coords, dtype=float) - np.asarray(origin)
    c, s = math.cos(heading), math.sin(heading)
    return points @ np.asarray([[c, -s], [s, c]])


def main():
    maps_db = get_maps_db(str(FLOW_ROOT / "data/maps"), "nuplan-maps-v1.0")
    map_api = NuPlanMapFactory(maps_db).build_map_from_name("sg-one-north")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=120)
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    for ax, (token, x, y, heading) in zip(axes.flat, CANDIDATES):
        nearby = map_api.get_proximal_map_objects(Point2D(x, y), 130, layers)
        polygons = []
        for layer in layers:
            for obj in nearby[layer]:
                polygons.append(transform(obj.polygon.exterior.coords, (x, y), heading))
        ax.add_collection(PolyCollection(polygons, facecolors="#edf1f3", edgecolors="#9aa4aa", linewidths=.6))
        ax.add_patch(plt.Rectangle((-2.5, -1.1), 5, 2.2, color="black", zorder=5))
        ax.arrow(0, 0, 3, 0, width=.12, head_width=.8, head_length=1, color="white", zorder=6)
        ax.set_xlim(-110, 110)
        ax.set_ylim(-110, 110)
        ax.set_aspect("equal")
        ax.set_title(token, fontsize=10)
        ax.grid(alpha=.15)
    fig.suptitle("Official interPlan nudge candidates, parked-car frame", fontsize=15)
    fig.tight_layout()
    output = HERE / "nudge_candidate_map_montage.png"
    fig.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
