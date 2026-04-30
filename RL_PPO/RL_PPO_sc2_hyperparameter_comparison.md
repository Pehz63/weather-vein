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
| Hidden layers | (64, 64) | (256, 256) | no | Mine uses 4x wider layers |
| # steps between updates | 2048 | 2400 (16 envs x 150 steps) | no | Sample counts are similar but structure differs - see below |
| Effective GAE horizon | 2048 | 150 | no | Paper's credit assignment spans up to 2048 steps; mine caps at 150 per episode |
| Batch size | 2048 | 2400 | ~ | Similar count, but mine is 16 independent 150-step episodes concatenated, not one long rollout |
| Minibatch size | 64 | 32 | no | Mine uses smaller minibatches |
| # minibatches | 32 | 75 (2400 / 32) | no | Mine produces more batches per epoch |
| Entropy bonus | not reported | 0.01 -> 0.001 (linear decay) | - | Not in paper; mine adds entropy regularization |
| Reward signal | raw farm power | differential (actual - zero-yaw) / 20.0 + terminal bonus + convergence penalty | no | Mine adds two extra shaping terms - see below |
| Network architecture | shared (implied) | split actor/critic backbones | - | Paper does not specify; mine keeps gradients separate |
| Observation space | wind + yaw (implied) | wind + yaw + normalized timestep t/T | - | Mine adds timestep so policy knows when to stop exploring |
| Episode length (Sc.2) | T = 2048 | T = 150 | no | Paper uses much longer episodes for Sc.2; mine uses 150 (the Sc.1 default) |
| Total training steps | 200k | ~480k (200 x 16 x 150) | ~ | Mine trains for ~2.4x more steps |

## Key Differences to Investigate

1. **Episode length / GAE horizon**: The paper uses T = 2048 for Scenario 2, but the environment is initialized with `max_num_steps=150`. This is the most significant structural mismatch. It's not just a sample-count difference - 16 x 150-step episodes have the same number of gradient steps as one 2400-step rollout, but GAE can only propagate credit within each 150-step episode. The value function never sees consequences beyond step 150, whereas the paper's policy learns to reason 2048 steps ahead. Longer episodes also let the agent observe the wind condition for longer before committing to a yaw strategy.

2. **Reward signal (differential + shaping vs raw)**: The paper trains on raw farm power. Mine uses three components: (1) differential reward `(actual - zero_yaw) / 20.0` to isolate the controllable yaw-steering gain; (2) a terminal bonus of `5x` the final-step differential, incentivizing the policy to find and hold the optimal yaw at episode end; (3) a convergence penalty `0.05 * (t/T)^2 * mean(|action|)` that grows quadratically through the episode, discouraging unnecessary yaw adjustments in the final steps.

3. **Observation space (12 vs 11 dims)**: A normalized timestep `t/T in [0, 1]` is appended to the observation vector. This lets the policy learn time-conditioned behavior - large yaw moves early, hold position late - which the shaping terms also encourage from the reward side.

4. **Hidden layer width (256 vs 64)**: Mine uses 4x wider hidden layers than the paper's (64, 64). The wider network has more capacity to represent the policy across the full wind condition space (speed 3-28 m/s, direction 0-360 deg).

5. **Total training steps (~480k vs 200k)**: Mine trains for ~2.4x more update cycles (200 vs the original 63), reaching ~480k total steps vs the paper's 200k. This gives the larger network and new reward signal more time to converge.

6. **Epochs per update (20 vs 10)**: More epochs risk overfitting to each rollout batch, but may speed learning. Could try halving to match the paper.

7. **Minibatch size (32 vs 64)**: Smaller minibatches mean noisier gradient estimates. Could double to match.
