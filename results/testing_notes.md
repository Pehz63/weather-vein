# Developer Testing Notes

## Overview

Three contributors evaluated wind farm yaw control methods using the FLORIS steady-state simulator via the `wfcrl` environment. All used two scenarios differing in whether the training wind conditions matched the fixed test conditions.

- **Scenario 1** — training wind = test wind (fixed, reproducible)
- **Scenario 2** — training wind is randomized; final test wind is fixed (same as Sc1)

---

## Wind Conditions at Evaluation (WIND_OPTS)

| Developer | Sc1 training wind | Sc1 **test** wind | Sc2 training wind | Sc2 **test** wind |
|-----------|-------------------|-------------------|-------------------|-------------------|
| **Kevin** | `wind_speed=8, wind_direction=270` (fixed) | `wind_speed=8, wind_direction=270` | Weibull(scale=8, shape=8) speed; Normal(μ=270, σ=20) direction | `wind_speed=8, wind_direction=270` (same as Sc1) |
| **Zeph** | `wind_speed=8, wind_direction=270` (fixed) | `wind_speed=8, wind_direction=270` | None → random per episode (Weibull speed, Normal direction) | `wind_speed=8, wind_direction=270` (fixed; corrected from random) |
| **aleksei** | `wind_speed=8, wind_direction=270` (fixed) | `wind_speed=8, wind_direction=270` | — | — (Sc2 not implemented) |

All three developers used `wind_direction=270°` (westerly, aligned perpendicular to turbine rows — noted as the challenging wake interaction case) and `wind_speed=8 m/s` for fixed conditions.

---

## Methodology Summary

| Aspect | Kevin | Zeph | aleksei |
|--------|-------|------|---------|
| **Methods tested** | GP, TabPFN, GraphPFN, Do-Nothing, Random | PPO (single agent per turbine) | GP, TabPFN, PFN4BO, KG-PFN, Axial PFN |
| **Simulator** | FLORIS (via `wfcrl`) | FLORIS (via `wfcrl`) | FLORIS (via `wfcrl`) |
| **Environments** | 20 FLORIS layouts (all registered) | 12 FLORIS layouts (FastFarm excluded) | Primarily `Turb3_Row1_Floris`; spot checks on others |
| **Episode length** | 150 steps | 150 steps | 150 steps |
| **Seeds / repetitions** | 3 seeds per layout per scenario | 10 eval episodes per env after training | 3 seeds (27, 28, 29) |
| **Baseline** | Do-Nothing (all yaws = 0) | No-control (zero yaw delta all steps) | Do-Nothing (all yaws = 0) |
| **Gain metric** | `(method_reward − do_nothing) / do_nothing × 100%` | `(ppo_mean − baseline_mean) / \|baseline_mean\| × 100%` | Absolute reward vs do-nothing total |
| **Scenarios completed** | Sc1 + Sc2 | Sc1 + Sc2 | Sc1 only |

---

## Method-Specific Details

| Developer | Method | Training data / episodes | Key hyperparameters |
|-----------|--------|--------------------------|---------------------|
| Kevin | GP | 32 context samples per run | RBF kernel, StandardScaler |
| Kevin | TabPFN | 32 context samples per run | TabPFN API (cloud inference) |
| Kevin | GraphPFN | 32 context samples per run | GAT v2, 2 layers × 2 heads; 1024 candidates searched |
| Kevin | Random | — | Uniform yaw ∈ [−40°, 40°] per turbine |
| Zeph | PPO | ~151 k steps per env (63 updates × 16 rollouts × 150 steps) | lr 3e-4→1e-5; clip ε=0.2; hidden 128 (Sc1) / 256 (Sc2); Sc2 warm-starts from Sc1 checkpoint |
| aleksei | GP | 500 simulation samples | Matern kernel (ν=0.5), ARD; Adam optimizer |
| aleksei | TabPFN | 1 000 simulation samples | Nelder-Mead optimization, 40 iterations |
| aleksei | PFN4BO / KG-PFN | 100 000 simulation samples | Transformer 512–1024 dim; 50-bucket discretized targets |

---

## Scenario 2 Notes

- **Kevin:** Scenario 2 tests whether surrogates trained on *variable-wind* context still optimize yaw under the standard fixed test condition. The 32 training samples span many (speed, direction) pairs; the final rollout always uses 8 m/s / 270°. This directly measures out-of-distribution generalization of the surrogate.

- **Zeph:** Scenario 2 initializes PPO policy from the Sc1 checkpoint (`LOAD_CHECKPOINT = True`) and continues training under random wind. Final evaluation uses fixed 8 m/s / 270° (corrected to match Kevin's protocol), so both developers' Sc2 numbers are comparable on the same test distribution.

- **aleksei:** Sc2 is acknowledged in `PFNs4WFCRL.ipynb` ("Sc. 2 samples the initial conditions randomly, so ensembling is a must") but no evaluation was executed.
