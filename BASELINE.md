# Flow Planner dynamics research baseline

This repository records the experiment orchestration, configurations, analysis,
and reports used for Flow Planner dynamics-conditioned expert-data research.
Large datasets, checkpoints, virtual environments, generated rollouts, and
external source checkouts are intentionally not committed.

## External source revisions

- Official Flow Planner: `366df7654d7b7fa77f9d2577279da63127c6d2f8`
- nuPlan devkit: `e9241677997dd86bfc0bcd44817ab04fe631405b`
- interPlan: `a85024a48935c92ec0d7c2a8cc2e583d01514469`
- Open Car Dynamics source snapshot: `94f8fb187fb0ed22bba1d809bd74f66d1ff75af4`
  (with per-file and submodule revisions recorded in
  `02_extreme_avoidance/upstream/manifest.json`)

At baseline creation, the Flow Planner checkout was detached at the revision
above and had no source modifications; only untracked Python bytecode caches
were present. The nuPlan and interPlan checkouts were clean.

## Frozen first-stage experiment

- Scenario: real nuPlan scenario `81f400b414215325-s0`.
- Expert replanning period: 0.1 s.
- LQR and OCD execution period: 0.1 s.
- Expert output: timestamped nuPlan reference trajectory; optimized controls
  are never applied directly to OCD.
- Closed-loop execution: the same official LQR and OCD chain for expert and
  Flow Planner references.
- Discriminating condition: initial longitudinal speed 12 m/s and OCD MF5.2
  internal peak scale 1.0 (an experiment parameter, not a calibrated friction
  coefficient).
- Verified outcome: the expert completed the pass without collision or road
  departure, with approximately 0.319 m minimum body clearance; the official
  Flow Planner collided at approximately 3.30 s under the same condition.

The detailed implementation and reproduction commands are documented in
`06_optimal_control_expert/README.md`.

## Scope of the next branch

The development branch created from this baseline is limited to scalable
expert-data collection: scenario parameterization, multiple expert maneuver
families, closed-loop validation, data contracts, resumable execution, and
dataset auditing. Dynamics-condition injection and supervised fine-tuning are
out of scope until the expert dataset is validated.
