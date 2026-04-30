# Hyperparameter Comparison: My PPO vs. WFCRL Paper (Table 5)

Paper reference: arXiv:2501.13592, Table 5 (IPPO/MAPPO hyperparameters)

| Parameter | Paper | Mine | Match? | Notes |
|---|---|---|---|---|
| Learning rate | 0.0003 -> 0 (linear) | 3e-4 -> 1e-5 (linear) | ~ | Paper decays to 0; mine floors at 1e-5 |
| beta (gamma) | 0.99 | 0.99 | yes | |
| GAE lambda | 0.95 | 0.95 | yes | |
| # epochs per update | 10 | 20 | no | Mine runs 2x as many gradient steps per rollout |
| Normalize advantages | True | True | yes | |
| Clip coefficient | 0.2 | 0.2 | yes | |
| Value loss coefficient | 0.5 | 0.5 | yes | |
| Max gradient norm | 0.5 | 0.5 | yes | |
| Hidden layers | (64, 64) | (128, 128) | no | Mine uses 2x wider layers |
| # steps between updates | 2048 | 2400 (16 envs x 150 steps) | no | Sample counts are similar but structure differs - see below |
| Effective GAE horizon | 2048 | 150 | no | Paper's credit assignment spans up to 2048 steps; mine caps at 150 per episode |
| Batch size | 2048 | 2400 | ~ | Similar count, but mine is 16 independent 150-step episodes concatenated, not one long rollout |
| Minibatch size | 64 | 32 | no | Mine uses smaller minibatches |
| # minibatches | 32 | 75 (2400 / 32) | no | Mine produces more batches per epoch |
| Entropy bonus | not reported | 0.01 -> 0.001 (linear decay) | - | Not in paper; mine adds entropy regularization |
| Reward signal | raw farm power | differential (actual - zero-yaw) / 20.0 | no | Major difference - see below |
| Network architecture | shared (implied) | split actor/critic backbones | - | Paper does not specify; mine keeps gradients separate |
| Episode length (Sc.2) | T = 2048 | T = 150 | no | Paper uses much longer episodes for Sc.2; mine uses 150 (the Sc.1 default) |
| Total training steps | 200k | ~151k (63 x 16 x 150) | ~ | Mine trains for slightly fewer steps |

## Key Differences to Investigate

1. **Episode length / GAE horizon**: The paper uses T = 2048 for Scenario 2, but the environment is initialized with `max_num_steps=150`. This is the most significant structural mismatch. It's not just a sample-count difference - 16 x 150-step episodes have the same number of gradient steps as one 2400-step rollout, but GAE can only propagate credit within each 150-step episode. The value function never sees consequences beyond step 150, whereas the paper's policy learns to reason 2048 steps ahead. Longer episodes also let the agent observe the wind condition for longer before committing to a yaw strategy.

2. **Reward signal (differential vs raw)**: The paper trains on raw farm power. Mine pre-computes zero-yaw baseline trajectories for every training seed and subtracts them step-by-step: `reward = (actual_power - zero_yaw_power) / 20.0`. This has two effects: (1) it reduces variance by removing the wind-condition contribution to the reward, so PPO only sees the yaw-steering delta; (2) it changes what the value function learns - the paper's critic must predict absolute power, mine predicts only the marginal gain from steering. This means the paper's policy must implicitly learn that high wind = high reward regardless of yaw, while mine learns to isolate the controllable component. A side effect is that if yaw steering offers no benefit, the reward is near zero even in strong wind, which could make learning harder or easier depending on the environment.

3. **Epochs per update (20 vs 10)**: More epochs risk overfitting to each rollout batch, but may speed learning. Could try halving to match the paper.

4. **Hidden layer width (128 vs 64)**: Larger network may help with the 11-dim observation space but adds parameters. The paper's (64, 64) is much smaller.

5. **Minibatch size (32 vs 64)**: Smaller minibatches mean noisier gradient estimates. Could double to match.
