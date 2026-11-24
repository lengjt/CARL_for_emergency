# Wide-lane static-obstacle closed-loop experiment

## Outcome

The replacement experiment ran end to end on the released Flow Planner weights.
The road has enough geometric space to pass on either adjacent lane, but the
planner did not initiate a lane change and did not add meaningful braking for the
barrier.  The ego collided with it at **4.699 s** while travelling **5.308 m/s**.
This is a model-behavior failure, not a program or data-pipeline failure.

No obstacle setting was changed after observing this result.

## Real scene and why it was selected

- nuPlan scenario token: `00513119374c5db4`
- log: `2021.06.09.17.23.18_veh-38_02526_03027`
- map: `us-nv-las-vegas-strip`
- nuPlan type: `unknown` (the token is a real extracted scene but has no named tag)
- initial speed: **8.388 m/s**
- sampled duration / step: **19.8 s / 0.1 s**
- current lane / roadblock: `67712` / `65685`
- adjacent lanes: `67730` and `67639`, in the same roadblock

Before either closed-loop run, the selector confirmed a nearly straight first
60 m, no lane connector along the logged path, and that roadblock `65685` is on
the original navigation route.  The two adjacent lane center separations are
3.542 m and 3.468 m.  After accounting for the 2.297 m ego width and 2.2 m
barrier width, the idealized parallel clearance margins are 1.294 m and 1.219 m.
Thus the scene permits a physical lane-change pass.

The 140 m local three-lane corridor contains seven moving tracks at the initial
frame.  It is not an empty synthetic road: logged vehicles later pass the
barrier location because surrounding traffic is deliberately non-reactive log
replay.  This is a limitation and is visible in the animations.

## Modification method

The scene edit follows interPlan's declarative per-token modification pattern,
recorded in `interplan_modification_wide_lane.yaml`.  The local executable
adapter is used instead of interPlan's native parked-vehicle modifier because
the latter creates a dynamic `VEHICLE`, while this experiment must exercise Flow
Planner's supported static `BARRIER` channel and preserve the original ego
history.  The real nuPlan map, route, ego history, and logged traffic are not
reconstructed or replaced.

The fixed barrier is centered 40.051 m along the expert path at global pose
`(664447.7718, 3996780.3309, -1.54106 rad)` with dimensions
`4.0 x 2.2 x 1.2 m`.  The same pose and dimensions are used by planner input,
collision tests, and rendering.

## Identical closed-loop settings

Both variants use seed 0, CFG weight 1.0, four flow ODE sample steps, the same
released checkpoint (SHA-256 begins `9231b4c0`), nuPlan LQR tracker, nuPlan
kinematic bicycle model, 0.1 s step, and non-reactive log replay.  At every step,
the executed bicycle-model state updates the history before Flow Planner is
called again.  Each run contains 199 planner calls and 198 executed controls.

## Vehicle behavior

Baseline: no collision.  The ego slows from 8.388 m/s to a minimum of 0.120 m/s
and later accelerates, following the model's response to the unmodified scene.

Barrier: first collision at 4.699 s and 5.308 m/s.  The minimum speed over the
full run is 0.112 m/s, but this later slowdown is almost identical to baseline
and is not successful obstacle braking.  Before collision, mean LQR acceleration
is -0.672 m/s² versus -0.662 m/s² in baseline.  At 4.60 s the speeds are 5.421
and 5.482 m/s respectively: only 0.061 m/s apart.

The future reference also shows no passing maneuver.  At planning times 0–4 s,
its center offset when it reaches the barrier is only about 0.00–0.18 m.  A
parallel pass requires at least 2.2485 m, so every sampled reference intersects
the barrier corridor.  The executed baseline and barrier paths consequently
remain almost coincident.

## Why seeing the barrier did not cause effective braking

The audit proves the barrier was present in all **199/199** actual model input
tensors; sorting, the five-static-object cap, and masking did not remove it.
Tensor-to-collision-geometry reconstruction error is 1.9e-6 m.  Therefore the
failure is not caused by a missing input.

Flow Planner produces one imitation-model trajectory sample here; this adapter
does not add a hard collision constraint, candidate rejection, or emergency
braking layer.  A visible static-category feature therefore does not guarantee
causal avoidance.  In this scene the model already commands substantial slowing
in the baseline, apparently from the broader logged context.  Adding the barrier
barely changes that profile and does not create a lane-change reference.  The
LQR then correctly tracks the unsafe reference; it is a tracker, not an
independent AEB system.  This also explains the earlier experiment: there the
unsafe reference maintained speed, so the LQR maintained speed too.  The two
scenes differ in background-induced speed profile, but both show weak response
to the injected static barrier.

## How to read the animations

Teal is the current eight-second Flow Planner future reference and updates every
0.1 s.  Blue is the accumulated bicycle-model trajectory actually executed so
far.  Red is the fixed added barrier, orange is non-reactive logged traffic, and
the dashed gray path is the logged human trajectory.  Divergence between teal
and blue is tracking/transient behavior; an unsafe teal line through red is a
planning failure before it is a tracking failure.

## Verification and limitations

All automated checks in `results_wide_lane/verification.json` pass: equal initial
state and timestamps, identical weights/seed/sampling, complete global reference
logs of shape `[199, 81, 5]`, barrier visibility, geometry consistency, and real
closed-loop control counts.

This remains one scene and one diffusion sample, not a benchmark.  Logged traffic
does not react to the added barrier or ego, the collision does not terminate the
simulation, and there is no external safety shield.  The experiment demonstrates
the released planner's behavior under this specific static-channel intervention;
it does not establish a general avoidance rate.
