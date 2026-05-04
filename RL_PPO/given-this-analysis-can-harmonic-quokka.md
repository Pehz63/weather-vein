# Plan: PPO Scenario 2 Improvements

## Context

The current PPO agent in `RL_PPO_Scenario_2.ipynb` achieves only ~0.7% gain over a no-control baseline (425.25 +/- 83.22 vs 422.29 +/- 85.86), far below the WFCRL paper target of 406.0 +/- 10.5 (which also has much lower variance, std=10.5 vs ours 83.22).

**Already implemented in the main notebook** (not to be re-added):
- Differential reward (`(r - baseline_step_reward) / REWARD_SCALE`) ✓
- 16 parallel envs ✓
- LR linear decay (3e-4 -> 1e-5) ✓
- NaN/gradient guards ✓

**Root causes that remain** (from analysis notes):
1. Agent has no sense of elapsed time - keeps adjusting yaw all 150 steps instead of converging and holding.
2. N_EPOCHS=20 is double the paper (10) - risks overfitting stale rollout data.
3. BATCH_SIZE=32 is half the paper (64) - noisier gradient estimates.
4. LR floors at 1e-5 rather than decaying to 0.

## Critical File

`RL_PPO_Scenario_2.ipynb` (main, not the worktree)

---

## Changes (in priority order)

### 1. Add normalized timestep to observation (highest impact)

**Cell:** `obs-utils`

- Modify `flatten_obs(obs)` to `flatten_obs(obs, step=0)`.
- Append `np.float32(step / 149.0)` as a 12th feature after the existing 11. This is already in [0,1] so no rescaling needed.
- Do the same in `flatten_obs_batch` - accept a `steps` array of shape (N,) and append it as a column.
- Update `OBS_DIM = 12`.
- Update `_OBS_LOW` and `_OBS_HIGH` to append 0.0 and 1.0 respectively (keeps the printout accurate).

**Cell:** `train-loop` - the vectorized loop tracks `step_idx = len(rew_bufs[i])` already for indexing `baseline_step_rewards`. Pass this as `steps=np.array([len(rew_bufs[i]) for i in range(N_ENVS)])` to `flatten_obs_batch`.

**Cell:** `eval-rollout` and `perf-tracking` - pass `step` to `flatten_obs(obs_dict, step)`.

**Why:** The analysis notes identify this as the primary driver of high variance. Without a timestep signal, the policy cannot distinguish "early, should search" from "late, should hold." The paper likely benefits from much longer episodes (2048 steps) giving the agent natural time awareness; at T=150 the agent never learns to converge.

---

### 2. Reduce N_EPOCHS from 20 to 10

**Cell:** `hparams`

```python
N_EPOCHS = 10   # was 20; matches WFCRL paper Table 5
```

**Why:** Running 20 gradient steps per rollout batch means the policy is updated far from the data distribution it was collected under. The PPO clip ratio (0.2) limits per-step change but 20 epochs compounds this risk. The paper uses 10. The current `ppo_update` does not track the approximate KL divergence as an early-stop signal, so it always runs all epochs even if the policy has drifted.

---

### 3. Increase BATCH_SIZE from 32 to 64

**Cell:** `hparams`

```python
BATCH_SIZE = 64   # was 32; matches paper Table 5
```

**Why:** Smaller minibatches give noisier gradient estimates. With 16 envs x 150 steps = 2400 samples per update, batch size 64 gives 37 minibatches per epoch (vs 75 at 32), which is still plenty while giving more stable gradients.

---

### 4. LR decay to 0 instead of 1e-5

**Cell:** `hparams`

```python
MIN_LR = 0.0   # was 1e-5
```

The LR decay logic in the train-loop already uses `max(LR * frac, MIN_LR)`. Changing `MIN_LR` to 0 is the only edit needed.

**Why:** Decaying fully to 0 is standard for PPO and ensures the policy stabilizes late in training rather than continuing to oscillate at a nonzero learning rate.

---

### 5. (Optional) Reduce hidden dim to 64

**Cell:** `actor-critic`

Change `hidden_dim=128` default to `hidden_dim=64`. Requires rebuilding the policy and re-running the baseline precomputation, so only try this if changes 1-4 still underperform.

**Why:** The paper uses (64, 64). With 12 input features, 128-wide layers are likely overkill. Smaller network = faster convergence on simple tasks and less overfitting risk.

---

## Verification

Run all cells top-to-bottom after changes 1-4. Key checkpoints:

- `obs_dim` prints `12` after `obs-utils`.
- Training reward printout values are still in the 300-500 range (raw power at eval, differential during training).
- `evaluate_policy` and `evaluate_no_control` both use raw rewards - compare their means directly.
- Target: mean eval reward closer to 406.0 and std much lower than 83.22.
- Run the `perf-tracking` cell (40 episodes) to check whether gain variance decreases across wind conditions.
