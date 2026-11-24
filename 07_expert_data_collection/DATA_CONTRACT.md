# Expert dataset contract v1

## Identity and grouping

- `condition_id` is a stable hash of scene, initial state, obstacle geometry,
  execution periods, tracker, and OCD environment parameters.
- All alternative solutions for exactly the same condition share one
  `condition_id`.
- `solution_id` combines the condition with one maneuver family. The pilot
  families are `safe_stop`, `pass_left`, and `pass_right`.
- Rejected solutions remain in the manifest for audit but are not listed as
  accepted training labels.

## Condition

`conditions/<condition_id>/condition.json` stores the real nuPlan scenario
identifier, initial ego state, obstacle pose and size, vehicle parameters,
tracker/execution periods, and the OCD MF5.2 internal peak scale. The scale is
explicitly not described as a calibrated road-friction coefficient.

## Reference labels

`solutions/<solution_id>/reference_labels.npz` contains only planning instants
where the optimizer returned a new successful solution. Reused references after
a failed solve are not duplicated as new labels.

- `query_state_global`: rear-axle pose and measured dynamic state at the query.
- `dynamics_condition`: longitudinal/lateral velocity, yaw rate, tire steering
  angle, and the internal MF5.2 peak scale.
- `reference_local`: future timestamped rear-axle reference in the query ego
  frame, with columns `relative_time_s, x_m, y_m, heading_rad, speed_mps`.
- `reference_global`: the same nuPlan reference in global coordinates for audit.

The reference is the training target. It is never replaced by the executed
trajectory.

## Closed-loop validation

`solutions/<solution_id>/executed_trajectory.npz` separately stores the actual
LQR + OCD trajectory and safety flags. `solution.json` records solver statistics,
collision, road departure, obstacle clearance, behavior completion, and the
admission decision.

For a side-specific pass, admission also requires the executed vehicle center
to cross the obstacle longitudinal station on the declared side. A side used
only as an optimizer initial guess does not determine the stored maneuver type.

The raw reproducibility logs are retained under `raw_runs/<solution_id>/`.
