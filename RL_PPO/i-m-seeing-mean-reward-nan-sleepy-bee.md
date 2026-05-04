# Plan: Fix mean reward=nan in 32-turbine training runs

## Context

Both `Turb_TCRWP_Floris` and `Turb32_Row5_Floris` (both 32-turbine, OBS_DIM=99) show
`mean reward=nan` and `pg=0.0000 vf=0.000000` every update. Smaller environments do not
exhibit this. The Turb32_Row5_Floris mid-rollout print at update 48 shows `reward mean=220.66`
with `done 0/16`, which means NaN arrives specifically at or after episode termination (step 150
truncation), not during the episode body.

## Root cause cascade

1. At the terminal step, the FLORIS simulator returns NaN power for the 32-turbine layout
   under certain wind conditions. Because `max_num_steps=150` triggers a hard truncation,
   every single episode ends with this NaN reward.

2. `r = float(np.squeeze(rewards[i]))` -> NaN
   - `ep_rewards[i] += r` -> NaN -> `ep_rewards.mean()` = NaN (the "mean reward=nan" line)
   - `diff_reward = (r - bl) / REWARD_SCALE` -> NaN -> appended to `rew_bufs[i]`

3. `compute_gae` back-propagates from the terminal NaN step:
   `delta = NaN + gamma * 0 - values[t]` -> NaN -> all 150 advantages become NaN.

4. `ppo_update` receives NaN advantages. `adv_t.mean()` = NaN ->
   normalized `adv_t` = NaN -> `loss` = NaN -> `if not torch.isfinite(loss): continue` ->
   every mini-batch skipped -> `n_batches == 0` -> returns `{"pg": 0.0, "vf": 0.0, "ent": 0.0}`.

Despite NaN training, `Turb_TCRWP_Floris` achieved +8.3% in evaluation because evaluation
uses different seeds / wind conditions that may not trigger the FLORIS NaN.

## Files to modify

- `RL_PPO/RL_PPO_Scenario_1_full.ipynb` - the training sweep cell (the large cell containing
  the rollout inner loop)

## Changes

### Fix 1 - Substitute baseline reward when environment returns NaN (rollout inner loop)

In the reward shaping block, after `r = float(np.squeeze(rewards[i]))`, add a guard before
diff_reward is computed. Use `bl` as the substitute so the shaped reward is exactly 0 (neutral
signal rather than destructive NaN):

```python
r = float(np.squeeze(rewards[i]))
ep_rewards[i] += r          # keep raw for reporting (shows NaN count visibly)

step_idx = len(rew_bufs[i])
bl = baseline_step_rewards[seeds[i]][step_idx]

# --- NEW: guard against NaN/inf from the simulator ---
r_for_shaping = r if np.isfinite(r) else bl   # diff_reward -> 0 when env returns NaN
if not np.isfinite(r):
    nan_reward_count += 1

diff_reward = (r_for_shaping - bl) / REWARD_SCALE
# ... convergence_penalty unchanged ...
if is_done:
    diff_reward += TERMINAL_SCALE * (r_for_shaping - bl) / REWARD_SCALE
```

`nan_reward_count` is reset to 0 at the top of each `for update in range(...)` iteration.

### Fix 2 - Replace ep_rewards display with nanmean

Change the two print_status / print calls that format `ep_rewards.mean()`:

```python
# rollout step print:
f"| reward mean={np.nanmean(ep_rewards):.2f}"

# force print:
f"| mean reward={np.nanmean(ep_rewards):.2f} "
```

Also in the EVAL_EVERY block:
```python
f"| mean reward: {np.nanmean(ep_rewards):.2f} "
```

This keeps `ep_rewards` raw (NaN values still stored, still appended to `episode_rewards`)
but shows a meaningful running metric instead of "nan".

### Fix 3 - Warn when NaN rewards are detected

At the end of each update (after `loss_log` append, before the per-update prints) add:

```python
if nan_reward_count > 0:
    print(f"  WARNING: {nan_reward_count} NaN rewards from env "
          f"(update {update+1}, {nan_reward_count}/{int(done_mask.sum()) * MAX_STEPS} steps)",
          flush=True)
```

This surfaces WHEN and HOW OFTEN the simulator produces NaN so the underlying FLORIS issue
can be tracked separately.

### Fix 4 - Sanitize critic values as defense in depth

After `action_t, logp_t, val_t = policy.act(obs_t)` and before the inner env loop,
ensure critic output doesn't propagate NaN into GAE:

```python
val_t = torch.nan_to_num(val_t, nan=0.0, posinf=0.0, neginf=0.0)
```

(logp_t would only be NaN if std -> 0, which is clamped; not strictly needed but keeps GAE clean.)

## What is NOT changed

- `compute_gae` - left intact; with Fix 1 in place the rewards it receives are finite
- `ppo_update` - the existing `isfinite` guard stays as a last resort; it should no longer
  trigger once rewards are clean
- `ActorCritic` class - no changes
- Hyperparameters / algorithm structure

## Verification

After applying the fixes, re-run one of the 32-turbine envs (e.g. `Turb32_Row5_Floris`).
Expected outcomes:
1. "mean reward" prints show numeric values (~200+) instead of "nan"
2. "pg" and "vf" losses show non-zero values (PPO updates are executing)
3. A WARNING line appears each update indicating how many NaN rewards were suppressed
4. The policy achieves similar or better evaluation gain than the current 8.3% result
