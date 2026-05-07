# region imports
# imports here...

# using device cuda

# Platform:    Windows-10-10.0.19045-SP0
# CPU cores:   12 logical, 6 physical
# RAM total:   25.6 GB
# RAM used:    19.1 GB  (74.5%)
# RAM avail:   6.5 GB
# Process RSS: 0.59 GB
# Swap used:   4.49 GB / 33.3 GB
# CPU%:        8.8%  (1-sec sample)


#region hyperparameters
# -- Actor-Critic -----------------------------------------------------------
HIDDEN_DIM  = 256
LOG_STD_MIN = -4.0
LOG_STD_MAX =  0.5


# -- Training budget -----------------------------------------------------------
ROLLOUTS_PER_UPDATE = 16  # episodes collected before each ppo_update call
N_EPISODES = 63          # update cycles (200 x 16 x 150 ~= 480K steps)
N_EVAL = 1              # number of runs when you do a final evaluation
EVAL_EVERY = 20           # print every N update cycles

# -- PPO update ----------------------------------------------------------------
LR              = 3e-4
MIN_LR          = 1e-5
N_EPOCHS        = 20
BATCH_SIZE      = 32
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
VF_COEF         = 0.5
ENT_COEF_START  = 0.01    # high entropy early -> exploration
ENT_COEF_END    = 0.001   # decay to low entropy late -> exploitation
ENT_COEF        = ENT_COEF_START  # alias for ppo_update default arg
MAX_GRAD        = 0.5

# Differential rewards are ~+-0.1-0.5 MW/step; scale so cumulative return is O(1)
REWARD_SCALE    = 20.0

# Terminal bonus: extra weight on the final-step differential reward.
# Encourages the agent to find the optimal yaw and be there at episode end.
TERMINAL_SCALE  = 5.0

# Convergence penalty: penalize large yaw deltas quadratically late in the episode.
# Penalty = CONVERGENCE_COEF * (t/T)^2 * mean(|action|), so early moves are free.
CONVERGENCE_COEF = 0.05

# NaN reward shaping:
# Replace NaN step rewards with a below-normal finite reward.
# Larger values punish NaNs more strongly. 1.0 = 1 std below mean;
# 2.0 = 2 std below mean, scaled further by the batch NaN rate.
NAN_REWARD_PENALTY_SCALE = 3.0
NAN_REWARD_MIN_STD = 1e-3

# Scenario 2: None = sample random wind speed and direction each episode
WIND_OPTS = None

# When doing cross-environment comparison, do only 1 eval on scenario 1 wind
STATIC_WIND_OPTS = {"wind_speed": 8, "wind_direction": 270}
STATIC_WIND_OPTS_FALLBACK = {"wind_speed": 8, "wind_direction": 270.01} # used upon NaN returns during final evaluation
CROSS_ENV_N_EVAL = 1
CROSS_ENV_FORCE_REEVAL = True  # set True to re-run eval even if already populated

# All 19 wfcrl environments. Fastfarm variants need an external FAST.Farm
# binary; if not installed, those iterations will fail and be recorded as
# errors in `results` rather than killing the whole sweep.
ENV_IDS = [
    # "Turb_TCRWP_Fastfarm",
    "Turb_TCRWP_Floris",
    # "Turb1_Row1_Fastfarm",
    "Turb1_Row1_Floris",
    # "Turb2_Row1_Fastfarm",
    "Turb2_Row1_Floris",
    # "Turb3_Row1_Fastfarm",
    "Turb3_Row1_Floris",
    "Turb10_Row1_Floris",
    # "Turb6_Row2_Fastfarm",
    "Turb6_Row2_Floris",
    # "Turb16_Row5_Fastfarm",
    "Turb16_Row5_Floris",
    # "Turb32_Row5_Fastfarm",
    "Turb32_Row5_Floris",
    "Ablaincourt_Floris",
    "HornsRev1_Floris",
    "HornsRev2_Floris",
    "Ormonde_Floris",
    "WMR_Floris",
]


N_ENVS = ROLLOUTS_PER_UPDATE
MAX_STEPS = 150  # max steps per episode (matches env max_num_steps)


HPARAMS_DICT = {
    "SEED": SEED, "N_EPISODES": N_EPISODES, "N_ENVS": N_ENVS,
    "LR": LR, "MIN_LR": MIN_LR, "N_EPOCHS": N_EPOCHS, "BATCH_SIZE": BATCH_SIZE,
    "GAMMA": GAMMA, "GAE_LAMBDA": GAE_LAMBDA, "CLIP_EPS": CLIP_EPS,
    "VF_COEF": VF_COEF, "ENT_COEF_START": ENT_COEF_START, "ENT_COEF_END": ENT_COEF_END,
    "MAX_GRAD": MAX_GRAD, "REWARD_SCALE": REWARD_SCALE,
    "TERMINAL_SCALE": TERMINAL_SCALE, "CONVERGENCE_COEF": CONVERGENCE_COEF,
    "MAX_STEPS": MAX_STEPS, "HIDDEN_DIM": HIDDEN_DIM,
    "LOG_STD_MIN": LOG_STD_MIN, "LOG_STD_MAX": LOG_STD_MAX,
}




#region Env setup
# Observation helpers. The sweep loop sets `_OBS_LOW`, `_OBS_HIGH`, `N_TURBINES`,
# `OBS_DIM`, `ACT_DIM`, and `env` per env_id; these helpers read whatever globals
# are currently in scope.
_OBS_KEYS = ["yaw", "freewind_measurements", "wind_speed", "wind_direction"]


def flatten_obs(obs: dict, step: int = 0, max_steps: int = MAX_STEPS) -> np.ndarray:
    """Convert Dict obs to a normalized flat vector in [-1, 1], appending normalized timestep.

    Raw features span very different scales (yaw: +-45, wind direction: 0-360).
    Normalizing to [-1, 1] prevents wind-direction values from dominating the
    first linear layer and keeps all gradients on the same scale.
    The timestep t/T in [0, 1] lets the policy distinguish early exploration
    from late-episode convergence.
    """
    raw = np.concatenate([obs[k] for k in _OBS_KEYS]).astype(np.float32)
    normalized = 2.0 * (raw - _OBS_LOW) / (_OBS_HIGH - _OBS_LOW) - 1.0
    t_norm = np.array([step / max_steps], dtype=np.float32)
    # Guard against NaN/inf from the simulator (e.g. async Floris I/O races).
    return np.nan_to_num(np.concatenate([normalized, t_norm]), nan=0.0, posinf=1.0, neginf=-1.0)


def flatten_obs_batch(obs_batch: dict, steps: np.ndarray = None, max_steps: int = MAX_STEPS) -> np.ndarray:
    """Vectorized version of flatten_obs. obs_batch[key] has shape (N, ...). Returns (N, OBS_DIM)."""
    raw = np.concatenate([obs_batch[k] for k in _OBS_KEYS], axis=-1).astype(np.float32)
    normalized = 2.0 * (raw - _OBS_LOW) / (_OBS_HIGH - _OBS_LOW) - 1.0
    N = normalized.shape[0]
    if steps is None:
        steps = np.zeros(N, dtype=np.float32)
    t_norms = (steps / max_steps).reshape(-1, 1).astype(np.float32)
    return np.nan_to_num(np.concatenate([normalized, t_norms], axis=-1), nan=0.0, posinf=1.0, neginf=-1.0)


def env_reset(seed_val, options):
    """Call env.reset() and always return just the obs dict. Reads global `env`."""
    result = env.reset(seed=seed_val, options=options)
    return result[0] if isinstance(result, tuple) else result



#region ActorCritic

class ActorCritic(nn.Module):
    """
    Split Actor-Critic for continuous actions.
    Separate backbones prevent VF loss gradients from corrupting actor weights.
    Actor: outputs a Gaussian (mean, log_std) over actions.
    Critic: outputs a scalar state value.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor_backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.critic_backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))  # learnable log-std
        self.critic        = nn.Linear(hidden_dim, 1)

        # Orthogonal init (standard for PPO)
        for backbone in (self.actor_backbone, self.critic_backbone):
            for layer in backbone:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _dist(self, obs: torch.Tensor) -> Normal:
        features = self.actor_backbone(obs)
        mean = self.actor_mean(features)
        std  = self.actor_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor):
        """Sample action; return (action, log_prob, value)."""
        actor_features  = self.actor_backbone(obs)
        critic_features = self.critic_backbone(obs)
        std  = self.actor_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        dist = Normal(self.actor_mean(actor_features), std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value    = self.critic(critic_features).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Re-evaluate stored (obs, action) pairs for the PPO loss."""
        actor_features  = self.actor_backbone(obs)
        critic_features = self.critic_backbone(obs)
        std  = self.actor_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        dist = Normal(self.actor_mean(actor_features), std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(critic_features).squeeze(-1)
        return log_prob, entropy, value


# region compute_gae
def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Generalized Advantage Estimation (GAE-lambda).
    Episode always terminates, so bootstrap value = 0.
    """
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae        = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        mask           = 1.0 - dones[t]
        delta          = rewards[t] + gamma * next_value * mask - values[t]
        gae            = delta + gamma * lam * mask * gae
        advantages[t]  = gae
        next_value     = values[t]
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns

# region ppo_update
def ppo_update(obs_buf, act_buf, logp_buf, adv_buf, ret_buf, ent_coef=ENT_COEF):
    """Mini-batch PPO update over N_EPOCHS passes."""
    obs_t  = torch.tensor(obs_buf,  dtype=torch.float32, device=DEVICE)
    act_t  = torch.tensor(act_buf,  dtype=torch.float32, device=DEVICE)
    logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=DEVICE)
    adv_t  = torch.tensor(adv_buf,  dtype=torch.float32, device=DEVICE)
    ret_t  = torch.tensor(ret_buf,  dtype=torch.float32, device=DEVICE)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)  # normalize advantages

    N = len(obs_t)
    total_pg = total_vf = total_ent = 0.0
    n_batches = 0

    for _ in range(N_EPOCHS):
        for b in torch.randperm(N, device=DEVICE).split(BATCH_SIZE):
            new_logp, entropy, new_val = policy.evaluate(obs_t[b], act_t[b])

            ratio    = (new_logp - logp_t[b]).exp()
            pg_loss  = -torch.min(
                ratio * adv_t[b],
                torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t[b]
            ).mean()
            vf_loss  = ((new_val - ret_t[b]) ** 2).mean()
            ent_loss = -entropy.mean()

            loss = pg_loss + VF_COEF * vf_loss + ent_coef * ent_loss
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                continue
            optimizer.step()
            optimizer.zero_grad()

            total_pg  += pg_loss.item()
            total_vf  += vf_loss.item()
            total_ent += ent_loss.item()
            n_batches += 1

    if n_batches == 0:
        return {"pg": 0.0, "vf": 0.0, "ent": 0.0}
    return {"pg": total_pg / n_batches,
            "vf": total_vf / n_batches,
            "ent": total_ent / n_batches}

# region eval policy
def evaluate_policy(n_episodes: int = 1, greedy: bool = True, seed_offset: int = 9001):
    """Return episode reward list and a DataFrame of the first episode's trajectory.
    Reads module globals: policy, env, env_reset, flatten_obs, STATIC_WIND_OPTS, N_TURBINES, DEVICE."""
    policy.eval()
    rewards   = []
    eval_rows = []

    with torch.no_grad():
        for ep in range(n_episodes):
            print_status(
                f"  eval policy | episode {ep+1}/{n_episodes} | starting",
                force=(ep == 0),
            )
            obs_dict = env_reset(seed_offset + ep, STATIC_WIND_OPTS)
            done, total_r, step = False, 0.0, 0

            while not done:
                obs_np   = flatten_obs(obs_dict, step=step)
                obs_t    = torch.tensor(obs_np, dtype=torch.float32,
                                        device=DEVICE).unsqueeze(0)
                if greedy:
                    features = policy.actor_backbone(obs_t)
                    action_t = policy.actor_mean(features)
                else:
                    action_t = policy.act(obs_t)[0]
                action_np    = np.clip(
                    action_t.squeeze(0).cpu().numpy(),
                    env.action_space["yaw"].low,
                    env.action_space["yaw"].high,
                )
                joint_action = {"yaw": action_np}

                obs_dict, reward, termination, truncation, info = env.step(joint_action)
                r        = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
                total_r += r
                print_status(
                    f"  eval policy | episode {ep+1}/{n_episodes} "
                    f"| step {step+1} | reward={r:.2f} | total={total_r:.2f}"
                )

                if ep == 0:
                    eval_rows.append({
                        "step":   step,
                        "reward": r,
                        **{f"yaw_{i}":   obs_dict["yaw"][i]   for i in range(N_TURBINES)},
                        **{f"power_{i}": info["power"][i]     for i in range(N_TURBINES)},
                    })
                step += 1
                done = termination or truncation

            rewards.append(total_r)
            print_status(
                f"  eval policy | episode {ep+1}/{n_episodes} complete "
                f"| steps={step} | total={total_r:.2f}",
                force=True,
            )

    policy.train()
    return rewards, pd.DataFrame(eval_rows)


# region eval no-op
def evaluate_no_control(n_episodes: int = 1, seed_offset: int = 9001):
    """Baseline: always send zero yaw deltas (no steering applied).
    Reads module globals: env, env_reset, WIND_OPTS, N_TURBINES."""
    zero_action = {"yaw": np.zeros(N_TURBINES, dtype=np.float32)}
    rewards = []
    baseline_rows = []

    for ep in range(n_episodes):
        print_status(
            f"  eval no-control | episode {ep+1}/{n_episodes} | starting",
            force=(ep == 0),
        )

        obs_dict = env_reset(seed_offset + ep, WIND_OPTS)
        done, total_r, step = False, 0.0, 0
        while not done:
            obs_dict, reward, termination, truncation, info = env.step(zero_action)
            r = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            total_r += r

            print_status(
                f"  eval no-control | episode {ep+1}/{n_episodes} "
                f"| step {step+1} | reward={r:.2f} | total={total_r:.2f}"
            )

            if ep == 0:
                baseline_rows.append({
                    "step": step,
                    "reward": r,
                    **{f"power_{i}": info["power"][i] for i in range(N_TURBINES)},
                })
            step += 1
            done = termination or truncation
        rewards.append(total_r)

        print_status(
            f"  eval no-control | episode {ep+1}/{n_episodes} complete "
            f"| steps={step} | total={total_r:.2f}",
            force=True,
        )

    print_status(
        f"  eval no-control complete | episodes={n_episodes} "
        f"| mean={np.mean(rewards):.2f} | std={np.std(rewards):.2f}",
        force=True,
    )

    return rewards, pd.DataFrame(baseline_rows)


# region Training Sweep
# ============================================================
# Training sweep across all env_ids
# ============================================================
# Per-env state is snapshotted into these dicts after each successful run so
# every trained policy stays in scope - the SELECTED_ENV cell below picks one
# to visualize, and the cross-env summary cells at the bottom plot all of them.

policies        = {}   # env_id -> trained ActorCritic
envs_dict       = {}   # env_id -> single-process eval env (kept open)
training_curves = {}   # env_id -> dict(episode_rewards, loss_log)
results         = {}   # env_id -> dict(eval_mean, eval_std, baseline_mean, gain_pct, dfs, ...)

for sweep_env_id in ENV_IDS:
    print(f"\n{'='*70}\nTraining: {sweep_env_id}\n{'='*70}")
    vec_env = None
    try:
        # ---- Single-process eval env -----------------------------------------
        env = envs.make(
            env_id=sweep_env_id,
            max_num_steps=150,
            controls={"yaw": (-45, 45, 5)},
            continuous_control=True,
            log=True,
        )
        N_TURBINES = env.num_turbines
        _OBS_LOW   = np.concatenate([env.observation_space[k].low  for k in _OBS_KEYS]).astype(np.float32)
        _OBS_HIGH  = np.concatenate([env.observation_space[k].high for k in _OBS_KEYS]).astype(np.float32)
        OBS_DIM    = _OBS_LOW.shape[0] + 1   # +1 for the appended timestep feature
        ACT_DIM    = N_TURBINES
        print(f"  N_TURBINES={N_TURBINES}, OBS_DIM={OBS_DIM}, ACT_DIM={ACT_DIM}")

        # ---- Fresh policy / optimizer ----------------------------------------
        policy    = ActorCritic(OBS_DIM, ACT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
        optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5)

        if LOAD_CHECKPOINT and sweep_env_id in CHECKPOINT_PATHS:
            SAVE_DIR = CHECKPOINT_PATHS[sweep_env_id]
            start_update, episode_rewards, loss_log = load_checkpoint(SAVE_DIR, policy, optimizer)
        else:
            episode_rewards = []
            loss_log = {"pg": [], "vf": [], "ent": []}
            start_update = 0
            SAVE_DIR = make_run_dir(sweep_env_id)
            print(f"  Saving to {SAVE_DIR}")

            # ---- Rollout env (sequential for Fastfarm to avoid MPI/spawn conflict)
            VecEnvCls = MpiSequentialVecEnv if "Fastfarm" in sweep_env_id else ParallelVecEnv
            vec_env = VecEnvCls([SEED + i for i in range(N_ENVS)], env_id=sweep_env_id)

            # ---- Per-env baseline cache (zero-yaw rewards per training seed) -----
            BASELINE_CACHE = os.path.join(SCENARIO_PATH_STR, sweep_env_id, f"baseline_cache_{sweep_env_id}_S{SEED}_E{N_EPISODES}_N{N_ENVS}.npz")
            if os.path.exists(BASELINE_CACHE):
                _data = np.load(BASELINE_CACHE)
                baseline_step_rewards = {int(k): list(_data[k]) for k in _data.files}
                print(f"  Loaded baseline cache ({len(baseline_step_rewards)} seeds)")
            else:
                baseline_step_rewards = {}
                _zero_action = {"yaw": np.zeros((N_ENVS, ACT_DIM), dtype=np.float32)}
                print(f"  Pre-computing baseline for {N_EPISODES * N_ENVS} seeds...")
                t0 = time.time()
                for _update in range(N_EPISODES):
                    seeds = [SEED + _update * N_ENVS + _i for _i in range(N_ENVS)]
                    obs_batch, _ = vec_env.reset(seed=seeds, options=WIND_OPTS)
                    step_rewards = [[] for _ in range(N_ENVS)]
                    dones = np.zeros(N_ENVS, dtype=bool)
                    baseline_step = 0
                    while not dones.all():
                        obs_batch, rewards, terminations, truncations, _ = vec_env.step(_zero_action)
                        for i in range(N_ENVS):
                            if not dones[i]:
                                step_rewards[i].append(float(np.squeeze(rewards[i])))
                                dones[i] = bool(terminations[i]) or bool(truncations[i])
                        baseline_step += 1
                        print_status(
                            f"    baseline batch {_update + 1}/{N_EPISODES} "
                            f"| step {baseline_step}/{MAX_STEPS} "
                            f"| done {dones.sum()}/{N_ENVS}"
                        )
                    for i, seed in enumerate(seeds):
                        baseline_step_rewards[seed] = step_rewards[i]
                    print_status(
                        f"    baseline {_update + 1}/{N_EPISODES} batches done "
                        f"({time.time()-t0:.1f}s)"
                    )
                print()
                os.makedirs(os.path.dirname(BASELINE_CACHE), exist_ok=True)
                np.savez(BASELINE_CACHE, **{str(k): np.array(v) for k, v in baseline_step_rewards.items()})
                print(f"  Saved baseline cache ({time.time()-t0:.1f}s)")

            # ---- Training loop ---------------------------------------------------
            ACT_LOW  = env.action_space["yaw"].low
            ACT_HIGH = env.action_space["yaw"].high
            remaining = N_EPISODES - start_update
            print(f"  Training: {remaining} updates x {N_ENVS} envs x {MAX_STEPS} steps "
                  f"~= {remaining * N_ENVS * MAX_STEPS:,} steps")

            for update in range(start_update, N_EPISODES):
                frac = 1.0 - update / N_EPISODES
                for pg in optimizer.param_groups:
                    pg["lr"] = max(LR * frac, MIN_LR)
                frac_ent = update / max(N_EPISODES - 1, 1)
                ent_coef = ENT_COEF_START + frac_ent * (ENT_COEF_END - ENT_COEF_START)

                obs_bufs  = [[] for _ in range(N_ENVS)]
                act_bufs  = [[] for _ in range(N_ENVS)]
                logp_bufs = [[] for _ in range(N_ENVS)]
                rew_bufs  = [[] for _ in range(N_ENVS)]
                val_bufs  = [[] for _ in range(N_ENVS)]
                done_bufs = [[] for _ in range(N_ENVS)]
                ep_rewards  = np.zeros(N_ENVS)
                env_had_nan = np.zeros(N_ENVS, dtype=bool)

                seeds = [SEED + update * N_ENVS + i for i in range(N_ENVS)]
                obs_batch, _ = vec_env.reset(seed=seeds, options=WIND_OPTS)
                done_mask     = np.zeros(N_ENVS, dtype=bool)
                step_counters = np.zeros(N_ENVS, dtype=np.int32)

                while not done_mask.all():
                    obs_np = flatten_obs_batch(obs_batch, steps=step_counters)
                    obs_t  = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
                    with torch.no_grad():
                        action_t, logp_t, val_t = policy.act(obs_t)
                    action_np = np.clip(action_t.cpu().numpy(), ACT_LOW, ACT_HIGH)
                    obs_batch, rewards, terminations, truncations, _ = vec_env.step({"yaw": action_np})

                    for i in range(N_ENVS):
                        if not done_mask[i]:
                            obs_bufs[i].append(obs_np[i])
                            act_bufs[i].append(action_np[i])
                            logp_bufs[i].append(logp_t[i].item())
                            r = float(np.squeeze(rewards[i]))
                            if not np.isfinite(r):
                                env_had_nan[i] = True
                            ep_rewards[i] += r

                            step_idx = len(rew_bufs[i])
                            bl = baseline_step_rewards[seeds[i]][step_idx]
                            diff_reward = (r - bl) / REWARD_SCALE

                            t_frac = step_idx / MAX_STEPS
                            convergence_penalty = CONVERGENCE_COEF * (t_frac ** 2) * float(np.mean(np.abs(action_np[i])))
                            diff_reward -= convergence_penalty

                            is_done = bool(terminations[i]) or bool(truncations[i])
                            if is_done:
                                diff_reward += TERMINAL_SCALE * (r - bl) / REWARD_SCALE

                            rew_bufs[i].append(diff_reward)
                            val_bufs[i].append(val_t[i].item())
                            done_bufs[i].append(float(is_done))
                            step_counters[i] += 1
                            if is_done:
                                done_mask[i] = True
                    print_status(
                        f"  Training {sweep_env_id} | update {update+1}/{N_EPISODES} "
                        f"| rollout step {int(step_counters.max())}/{MAX_STEPS} "
                        f"| done {done_mask.sum()}/{N_ENVS} "
                        f"| reward mean={np.nanmean(ep_rewards):.2f}"
                    )

                obs_all, act_all, logp_all, adv_all, ret_all = [], [], [], [], []
                n_nan_eps = int(env_had_nan.sum())
                if n_nan_eps > 0:
                    print(f"  [NaN] Discarding {n_nan_eps}/{N_ENVS} episodes "
                          f"(FLORIS instability, update {update+1})", flush=True)
                for i in range(N_ENVS):
                    if env_had_nan[i]:
                        continue
                    adv, ret = compute_gae(rew_bufs[i], val_bufs[i], done_bufs[i])
                    obs_all.append(np.array(obs_bufs[i]))
                    act_all.append(np.array(act_bufs[i]))
                    logp_all.append(np.array(logp_bufs[i]))
                    adv_all.append(adv)
                    ret_all.append(ret)
                    episode_rewards.append(ep_rewards[i])

                print_status(
                    f"  Training {sweep_env_id} | update {update+1}/{N_EPISODES} "
                    f"| rollout done, running PPO update"
                )
                losses = {"pg": float("nan"), "vf": float("nan"), "ent": float("nan")}
                if len(obs_all) == 0:
                    print(f"  [NaN] All {N_ENVS} episodes had NaN -- skipping PPO update "
                          f"(update {update+1})", flush=True)
                else:
                    losses = ppo_update(
                        np.concatenate(obs_all), np.concatenate(act_all),
                        np.concatenate(logp_all), np.concatenate(adv_all),
                        np.concatenate(ret_all), ent_coef=ent_coef,
                    )
                for k, v in losses.items():
                    loss_log[k].append(v)
                
                print_status(
                    f"  Training {sweep_env_id} | update {update+1}/{N_EPISODES} "
                    f"| mean reward={np.nanmean(ep_rewards):.2f} "
                    f"| pg={losses['pg']:.4f} vf={losses['vf']:.6f}",
                    force=True,
                    newline=((update + 1) % EVAL_EVERY == 0),
                )
                if (update + 1) % EVAL_EVERY == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(f"  Update {update+1:3d}/{N_EPISODES} | lr={cur_lr:.2e} ent={ent_coef:.4f} "
                          f"| mean reward: {np.nanmean(ep_rewards):.2f} "
                          f"| pg={losses['pg']:.4f}  vf={losses['vf']:.6f}")

                if (update + 1) % 10 == 0:
                    print_status(
                        f"  Training {sweep_env_id} | update {update+1}/{N_EPISODES} "
                        f"| saving checkpoint"
                    )
                    save_checkpoint(SAVE_DIR, policy, optimizer, update + 1,
                                    episode_rewards, loss_log, HPARAMS_DICT)

            save_checkpoint(SAVE_DIR, policy, optimizer, N_EPISODES,
                            episode_rewards, loss_log, HPARAMS_DICT)

            vec_env.close()
            vec_env = None

        # ---- Snapshot (always) -----------------------------------------------
        policies[sweep_env_id]        = policy
        envs_dict[sweep_env_id]       = env
        training_curves[sweep_env_id] = {"episode_rewards": list(episode_rewards),
                                         "loss_log": {k: list(v) for k, v in loss_log.items()}}

        # ---- Evaluation ------------------------------------------------------
        if RUN_EVAL:
            n_eval = N_EVAL
            print_status(
                f"  Evaluating {sweep_env_id} | PPO policy | episodes={n_eval}",
                force=True, newline=True
            )
            eval_rewards, eval_df = evaluate_policy(n_episodes=n_eval)
            print_status(
                f"  Evaluating {sweep_env_id} | no-control baseline "
                f"| episodes={max(N_EVAL, n_eval // 3)}",
                force=True, newline=True
            )
            baseline_rewards, baseline_df = evaluate_no_control(n_episodes=max(N_EVAL, n_eval // 3))
            eval_df["total_power"]     = sum(eval_df[f"power_{i}"]     for i in range(N_TURBINES))
            baseline_df["total_power"] = sum(baseline_df[f"power_{i}"] for i in range(N_TURBINES))
            results[sweep_env_id] = {
                "eval_rewards":    eval_rewards,
                "eval_df":         eval_df,
                "baseline_rewards": baseline_rewards,
                "baseline_df":     baseline_df,
                "eval_mean":       float(np.mean(eval_rewards)),
                "eval_std":        float(np.std(eval_rewards)),
                "baseline_mean":   float(np.mean(baseline_rewards)),
                "baseline_std":    float(np.std(baseline_rewards)),
                "gain_pct":        100.0 * (np.mean(eval_rewards) - np.mean(baseline_rewards))
                                   / abs(np.mean(baseline_rewards) + 1e-9),
                "n_turbines":      N_TURBINES,
                "obs_low":         _OBS_LOW.copy(),
                "obs_high":        _OBS_HIGH.copy(),
                "save_dir":        SAVE_DIR,
            }
            print(f"  DONE: PPO={results[sweep_env_id]['eval_mean']:.2f} +- "
                  f"{results[sweep_env_id]['eval_std']:.2f}, "
                  f"baseline={results[sweep_env_id]['baseline_mean']:.2f}, "
                  f"gain={results[sweep_env_id]['gain_pct']:+.1f}%")
        else:
            results[sweep_env_id] = {
                "eval_rewards":    [],
                "eval_df":         None,
                "baseline_rewards": [],
                "baseline_df":     None,
                "eval_mean":       float("nan"),
                "eval_std":        float("nan"),
                "baseline_mean":   float("nan"),
                "baseline_std":    float("nan"),
                "gain_pct":        float("nan"),
                "n_turbines":      N_TURBINES,
                "obs_low":         _OBS_LOW.copy(),
                "obs_high":        _OBS_HIGH.copy(),
                "save_dir":        SAVE_DIR,
            }
            print(f"  Skipped evaluation (RUN_EVAL=False)")

    except Exception as exc:
        print(f"  FAILED ({type(exc).__name__}): {exc}")
        results[sweep_env_id] = {"error": f"{type(exc).__name__}: {exc}"}
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:
                pass

print(f"\n{'='*70}\nSweep complete\n{'='*70}")
n_ok = sum(1 for r in results.values() if "error" not in r)
print(f"  {n_ok}/{len(ENV_IDS)} envs trained successfully")
for env_id_, r in results.items():
    if "error" in r:
        print(f"  {env_id_}: FAILED - {r['error']}")
    else:
        print(f"  {env_id_}: PPO={r['eval_mean']:.2f}+-{r['eval_std']:.2f}, "
              f"baseline={r['baseline_mean']:.2f}, gain={r['gain_pct']:+.1f}%")


