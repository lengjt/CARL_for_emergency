import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "02_extreme_avoidance"))

from cfm_model.numpy_model import NumpyCFM
from cfm_model.scenarios import build_scenarios
from extreme_avoidance.scenario import Obstacle, Road, Scenario


class CoreTests(unittest.TestCase):
    def test_family_split_keeps_variants_together(self):
        config = json.loads((ROOT / "configs/data.json").read_text())
        rows = build_scenarios(config, (Scenario, Road, Obstacle))
        self.assertEqual(len(rows), 90)
        by_family = {}
        for row in rows:
            by_family.setdefault(row["family_id"], set()).add(row["split"])
        self.assertEqual(len(by_family), 18)
        self.assertTrue(all(len(value) == 1 for value in by_family.values()))
        self.assertEqual({split: len({row["family_id"] for row in rows if row["split"] == split})
                          for split in ("train", "val", "test")},
                         {"train": 12, "val": 3, "test": 3})

    def test_numpy_sampler_is_seed_reproducible(self):
        width, condition_dim, target_dim = 4, 12, 3
        arrays = {
            "input__weight": np.zeros((width, condition_dim + target_dim + 3)),
            "input__bias": np.zeros(width),
            "blocks__0__norm__weight": np.ones(width),
            "blocks__0__norm__bias": np.zeros(width),
            "blocks__0__linear1__weight": np.zeros((width, width)),
            "blocks__0__linear1__bias": np.zeros(width),
            "blocks__0__linear2__weight": np.zeros((width, width)),
            "blocks__0__linear2__bias": np.zeros(width),
            "output_norm__weight": np.ones(width),
            "output_norm__bias": np.zeros(width),
            "output__weight": np.zeros((target_dim, width)),
            "output__bias": np.zeros(target_dim),
        }
        metadata = {"condition_mean": [0] * condition_dim, "condition_std": [1] * condition_dim,
                    "target_mean": [0, 20, 1], "target_std": [1, 5, 1],
                    "target_clip": [[-4, 4], [5, 60], [0, 5]], "residual_blocks": 1}
        arrays["metadata_json"] = np.asarray([json.dumps(metadata)], dtype=object)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            np.savez(path, **arrays)
            model = NumpyCFM(path)
            first = model.sample(np.zeros(condition_dim), 5, 2, 7)
            second = model.sample(np.zeros(condition_dim), 5, 2, 7)
            np.testing.assert_equal(first, second)
            self.assertTrue(np.all(first[:, 2] >= 0))


if __name__ == "__main__":
    unittest.main()
