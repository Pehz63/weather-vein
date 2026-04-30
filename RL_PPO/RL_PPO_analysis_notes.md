# RL PPO Scenario 2 - Analysis Notes

## Inputs and Controls

**Observation (11 features), normalized to [-1, 1]:**

| Index | Feature | Shape | Range |
|-------|---------|-------|-------|
| 0-2 | Current yaw per turbine | (3,) | -45 to +45 deg |
| 3-4 | Freewind speed + direction | (2,) | 3-28 m/s, 0-360 deg |
| 5-7 | Wind speed per turbine | (3,) | 3-28 m/s |
| 8-10 | Wind direction per turbine | (3,) | 0-360 deg |

**Action:** yaw delta per turbine, [-5, +5] degrees per step, accumulated into absolute yaw bounded at [-45, +45].

**Model:** Two independent 2-layer Tanh MLPs (128 hidden), one for the actor (outputs Gaussian mean + learned log-std), one for the critic (outputs scalar value). Separate backbones prevent value-function gradients from corrupting the policy.

---

## What is Randomized

Wind is sampled **once per episode** at `reset()`:

- **Wind speed:** `8 * Weibull(shape=8)`, clipped to [3, 28] m/s - peaks around 8 m/s
- **Wind direction:** `Normal(270, 20) % 360`, clipped to [0, 360] - centered at 270 deg (west, aligned with the turbine row)

Wind is **constant within an episode** - FLORIS is a steady-state solver, not time-varying.

---

## What the Simulator Does

FLORIS is a **steady-state wake model** - it does not simulate time-evolving physics. Each `step()` call:

1. Applies the yaw delta (clipped to action bounds, accumulated into absolute yaw bounded at [-45, 45])
2. Calls FLORIS to compute a new steady-state power solution for the current yaw configuration
3. Returns total farm power as the reward

There is no inertia, no wind dynamics, no transients. The 150-step episode exists purely to let the agent incrementally steer yaw angles toward a good configuration (+/-5 deg per step max, so reaching -45 from 0 takes at least 9 steps).

---

## Does the Model Know Current Timestep or Current Yaw?

- **Current yaw:** YES - obs[0:3] are the absolute yaw angles of each turbine. The agent always knows where it is.
- **Current timestep:** NO - there is no step index in the observation. The agent has no built-in sense of urgency or how much time remains.

---

## Wind Farm Layout

3 turbines in a row at 4D spacing (D = 126 m rotor diameter):

- T0: x=0 m, y=0 m (most upstream when wind is from 270 deg)
- T1: x=504 m, y=0 m
- T2: x=1008 m, y=0 m (most downstream)

Wind direction 270 deg = wind comes from the west, flows east (+x direction), directly aligned with the row. This is the mean of the sampled distribution, so wake effects are strongest on average.

---

## Performance Patterns

From the 20-episode eval:

| Episode | PPO | Baseline | Gain |
|---------|-----|----------|------|
| 8 | 244 | 229 | +6.5% |
| 13 | 283 | 251 | +12.5% |
| 5 | 349 | 358 | -2.8% |
| 17 | 237 | 267 | -11.1% |
| 0, 3, 10, ... | ~490 | ~491 | ~0% |

**Overall:** Mean 425.25 +/- 83.22 vs baseline 422.29 +/- 85.86. Only +0.7% gain, far below WFCRL IPPO target of 406.0 +/- 10.5.

**Pattern - when baseline is ~490 (high wind, ~270 deg direction):**
The farm is already nearly at max power with zero yaw. No room to improve. PPO sometimes slightly hurts by wandering away from zero during the episode.

**Pattern - when baseline is low (~230-280, low wind speed or off-axis direction):**
Results are mixed. PPO can help a lot (+12%) or hurt a lot (-11%). The policy has not robustly learned to exploit those conditions.

**Root cause of high variance (std 83 vs WFCRL's 10.5):**

1. The model does not know the timestep, so it keeps adjusting yaw all 150 steps rather than converging quickly. Early steps pay an "exploration tax" vs the baseline which is already at stable zero yaw.
2. The policy cannot distinguish "I'm early and should search" from "I've found the optimum and should hold."
3. The differential reward (PPO power - zero-yaw power) removes wind-condition variance during training, but at eval time the absolute reward variance is still dominated by which wind conditions get sampled.

---

## New Notebook Cells Added

Cells appended to `RL_PPO_Scenario_2.ipynb` (starting at cell 34):

- **farm-layout** - draws the 3-turbine row with wind arrows, rotor-plane lines, and yaw-offset thrust arrows; shows step 0 vs step 148 for episode 0
- **yaw-animation** - `FuncAnimation` showing farm layout updating as yaw evolves, with live yaw-vs-step plot side-by-side
- **perf-tracking** - re-runs 40 episodes tracking wind speed, wind direction, PPO reward, and baseline reward per episode
- **perf-plot** - 3-panel scatter: gain vs wind direction, gain vs wind speed, gain sorted by episode difficulty
