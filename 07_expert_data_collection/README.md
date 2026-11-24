# Small-scale expert data collector

This stage freezes the first expert-data contract and validates a three-candidate
pilot on the existing nuPlan scene. It does not modify Flow Planner tokens,
inject dynamics conditions, or train a model.

The pilot evaluates `safe_stop`, `pass_left`, and `pass_right` independently.
Every candidate produces timestamped nuPlan references and is executed through
the same official LQR and OCD chain. A candidate is admitted only after the OCD
closed loop is collision-free, remains within the route polygon, and completes
its declared behavior (stop or pass).

Generated output lives under `outputs/pilot/` and is intentionally excluded
from Git. The fixed schema is documented in `DATA_CONTRACT.md`.

```bash
cd /home/ljt23/research_ca
01_flow_planner/.venv/bin/python 07_expert_data_collection/collect_pilot.py
01_flow_planner/.venv/bin/python 07_expert_data_collection/verify_dataset.py
01_flow_planner/.venv/bin/python 07_expert_data_collection/summarize_pilot.py
```
