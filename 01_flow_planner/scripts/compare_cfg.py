"""Check same-seed equivalence at guidance weight 1; no training, B=1 only."""
import json
from common import ROOT, load_model
from run_inference import synthetic_scene
import torch


def main():
    torch.set_num_threads(4)
    model, _ = load_model(cfg_weight=1.0)
    data = synthetic_scene().to("cuda")
    errors = []
    with torch.inference_mode():
        for seed in (3402, 3403, 3404, 3405, 3406):
            torch.manual_seed(seed)
            doubled = model(data, mode="inference", use_cfg=True, cfg_weight=1.0)
            torch.manual_seed(seed)
            single = model(data, mode="inference", use_cfg=False, cfg_weight=1.0)
            errors.append(float((doubled - single).abs().max()))
            torch.testing.assert_close(doubled, single, atol=1e-3, rtol=1e-4)
    report = {"input_kind": "synthetic_interface_fixture", "seeds": [3402, 3403, 3404, 3405, 3406],
              "cfg_weight": 1.0, "max_abs_errors": errors,
              "test": "CFG doubled branch vs single conditioned branch, identical seeds",
              "tolerance": {"atol": 1e-3, "rtol": 1e-4},
              "passed": True, "scope": "Only this fixture and these seeds; not a real-data behavior evaluation"}
    (ROOT / "artifacts/cfg_equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
