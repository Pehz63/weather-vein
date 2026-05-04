# Plan: Principled Improvement Strategy for PPO Wind Farm Yaw Control

## Context

The current PPO agent achieves only +0.7% gain over zero-yaw baseline. Five changes were implemented this session (timestep obs, convergence penalty, terminal bonus, wider network, more training), but the specific parameter values - `TERMINAL_SCALE=5.0`, `CONVERGENCE_COEF=0.05`, `hidden_dim=256` - were chosen by reasoning, not measurement. Without a structured approach, a failed run leaves us with no actionable information.

This plan replaces guess-and-retry with three sequenced experiments, each gated on the result of the previous one. Each is implemented as a new cell (or section) in `RL_PPO_Scenario_2.ipynb`.

**Key timing data** (from notebook output):
- Baseline precomputation: 1275s for 1008 seeds = **1.27s/seed**
- Training: ~9 min for 63 update cycles = **8.6s/update**
- Original full run (N_EPISODES=63): **~30 min**
- Current full run (N_EPISODES=200): **~97 min** (68 min precompute + 29 min train)

---

## Step 1: FLORIS Oracle (~10-20 min)

**Why first:** Cheapest experiment, answers the most important question before spending hours on training. If the true optimum is only +2% above zero-yaw, the model is already near the ceiling and Steps 2-3 are unnecessary.

**What it does:** Uses `scipy.optimize` to find the globally optimal static yaw angles for each eval seed by calling FLORIS directly - one simulator call per yaw configuration, no need for full 150-step episodes.

**Runtime math:**
- FLORIS single-step cost: 1275s / (1008 seeds x 150 steps) = ~8.4ms/step
- Per eval seed: ~200 optimizer calls x ~17ms (reset + 1 step) = ~3.4s
- 20 eval seeds: ~70s compute + overhead = **10-20 min total**

**Implementation** - add a new cell after the eval section in `RL_PPO_Scenario_2.ipynb`:

```python
from scipy.optimize import minimize

YAW_BOUNDS = [(-45, 45)] * N_TURBINES

def floris_power(yaw_angles, seed):
    """Single FLORIS query: reset to seed's wind, apply yaw, return farm power."""
    obs = env_reset(seed, WIND_OPTS)
    action = {"yaw": np.clip(yaw_angles, -45, 45).astype(np.float32)}
    # One env step with fixed yaw gives steady-state power for this wind condition
    _, reward, _, _, _ = env.step(action)
    return -(float(reward[0]) if hasattr(reward, "__len__") else float(reward))

oracle_rewards = []
for ep in range(20):
    seed = 9001 + ep
    result = minimize(
        floris_power, x0=np.zeros(N_TURBINES),
        args=(seed,), method="Nelder-Mead",
        bounds=YAW_BOUNDS,
        options={"maxiter": 300, "xatol": 0.5, "fatol": 0.1},
    )
    oracle_rewards.append(-result.fun)

print(f"Oracle mean +- std:   {np.mean(oracle_rewards):.2f} +- {np.std(oracle_rewards):.2f}")
print(f"PPO mean +- std:      {np.mean(eval_rewards):.2f} +- {np.std(eval_rewards):.2f}")
print(f"Zero-yaw mean:        {np.mean(baseline_rewards):.2f}")
print(f"PPO gap vs oracle:    {np.mean(oracle_rewards) - np.mean(eval_rewards):.2f} MW")
print(f"Headroom above baseline: {np.mean(oracle_rewards) - np.mean(baseline_rewards):.2f} MW "
      f"({(np.mean(oracle_rewards)/np.mean(baseline_rewards)-1)*100:+.1f}%)")
```

**Gate:** If oracle headroom is < 5 MW above zero-yaw baseline, stop - the environment offers little room to improve and the model is already near-optimal. If >= 5 MW, proceed to Step 2.

---

## Step 2: Ablation Studies (~4.5 hours)

**Why second:** Before searching over parameter values, confirm which of the 5 changes actually help. Without ablations, a failed combined run is undiagnosable. Running ablations at N_EPISODES=63 (original budget) keeps each run at ~30 min.

**Runtime math:**
- Each run at N_EPISODES=63: ~30 min
- 5 isolated runs + 1 baseline + 1 full combined (N_EPISODES=200) = 5x30 + 30 + 97 = **~4.5 hours**

**Implementation** - a separate helper function that rebuilds the policy and runs training with a configurable set of flags, called once per ablation config:

```python
def run_ablation(name, use_timestep=False, use_conv_penalty=False,
                 use_terminal=False, hidden_dim=128, n_episodes=63):
    """Train from scratch with a specific subset of changes. Returns eval mean +- std."""
    ...
```

**Ablation matrix** (each row is one training run):

| Run | Timestep obs | Conv penalty | Terminal bonus | hidden_dim | N_eps | Est. time |
|-----|---|---|---|---|---|---|
| Baseline | - | - | - | 128 | 63 | 30 min |
| +timestep | X | - | - | 128 | 63 | 30 min |
| +conv penalty | - | X | - | 128 | 63 | 30 min |
| +terminal bonus | - | - | X | 128 | 63 | 30 min |
| +hidden 256 | - | - | - | 256 | 63 | 30 min |
| All combined | X | X | X | 256 | 200 | 97 min |
| **Total** | | | | | | **~4.5 hours** |

**Gate:** Any change that does not improve mean reward OR increases std relative to the baseline run should be dropped before Step 3. Only proceed to Optuna for parameters whose changes demonstrably mattered.

---

## Step 3: Optuna Search (~9 hours, only if Steps 1-2 justify it)

**Why last:** Most expensive by far. Only worth running once ablations have narrowed which parameters matter, collapsing the search space.

**Runtime math (proxy strategy):**
- Full run at N_EPISODES=63: 30 min/trial -> 30 trials = 15 hours (too slow)
- Proxy run at N_EPISODES=30: ~14 min/trial (607s precompute + 4 min train)
- 30 proxy trials + 1 final full run at N_EPISODES=200: 30x14 + 97 = **~8 hours**

**Search space** (restricted to parameters ablations identified as impactful):

```python
import optuna

def objective(trial):
    terminal_scale   = trial.suggest_float("terminal_scale",   0.5, 10.0)
    convergence_coef = trial.suggest_float("convergence_coef", 0.01, 0.3)
    hidden_dim       = trial.suggest_categorical("hidden_dim", [128, 256])
    # Run training at N_EPISODES=30 as a proxy for full performance
    mean_reward, _ = run_ablation(..., n_episodes=30)
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(study.best_params)
```

After search, do one final full training run (N_EPISODES=200) with the best found parameters to get the definitive result.

---

## Recommended Execution Order

```
Run oracle (10-20 min)
  -> headroom < 5 MW? STOP, model is already near-optimal
  -> headroom >= 5 MW? Run ablations (4.5 hours)
      -> no single change helps? Revisit reward/architecture assumptions
      -> some changes help? Run Optuna on those params only (~8 hours)
          -> Final full run with best params (97 min)
```

**Total worst-case:** ~14 hours across multiple sessions
**Likely path:** Oracle + ablations only (~5 hours), Optuna only if ablations are inconclusive on values

---

## Files Modified

- `RL_PPO_Scenario_2.ipynb` - new cells added after the existing eval section for oracle, ablation runner, and Optuna objective

## Verification

After each step, compare against the three benchmarks already in the notebook:
- Zero-yaw baseline: 422.29 +- 85.86 MW
- Current PPO: 425.25 +- 83.22 MW
- WFCRL IPPO target: 406.0 +- 10.5 MW (note: lower mean but dramatically lower std)

A meaningful improvement means both higher mean AND lower std - the WFCRL target beats our current mean by 19 MW but has 8x lower variance, indicating a much more consistent policy.
