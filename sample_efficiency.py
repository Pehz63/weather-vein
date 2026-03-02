import argparse
import os
import io
import contextlib
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

import torch
from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
from botorch.models import MultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel

from tabpfn import TabPFNRegressor
from wfcrl import environments as envs

from pfns4bo import priors, encoders, utils, bar_distribution, train as pfns_train
from pfns4bo.priors.utils import Batch

from src import step_policy, obs_to_row

import warnings
warnings.filterwarnings("ignore")

for var in ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK',
            'SLURM_PROCID', 'SLURM_NTASKS', 'SLURM_NODELIST']:
    os.environ.pop(var, None)
    
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=27)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_steps', type=int, default=150)
parser.add_argument('--n_seeds', type=int, default=3, help="Seeds for error bars")
args = parser.parse_args()

OPTIONS = {"wind_speed": 8, "wind_direction": 270}
LAYOUTS = ['Turb3_Row1_Floris', 'Ablaincourt_Floris']

SAMPLE_BUDGETS = [150, 250, 500, 1000, 5000]
GP_MAX_SAMPLES = 500


def make_env(layout):
    return envs.make(
        layout,
        max_num_steps=args.max_steps,
        controls={"yaw": (-5, 5)},
        continuous_control=True,
        log=True,
    )


def collect_sim_data(env, n_rows, seed):
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    while len(rows) < n_rows:
        obs = env.reset(seed=seed, options=OPTIONS)
        i, done = 0, False
        while not done:
            action = step_policy(i, env)
            obs, reward, term, trunc, info = env.step(action)
            rows.append(obs_to_row(obs, reward, info["power"], i))
            i += 1
            done = term or trunc
    return pd.DataFrame(rows[:n_rows])


def eval_do_nothing(env):
    env.reset(options=OPTIONS)
    r = 0
    for _ in range(args.max_steps):
        _, reward, _, _, _ = env.step({'yaw': np.zeros(env.num_turbines)})
        r += reward
    return r[0]


def ramp_and_eval(env, target_yaws, seed):
    obs = env.reset(seed=seed, options=OPTIONS)
    r, done = 0, False
    while not done:
        delta = np.clip(target_yaws - obs['yaw'], -5, 5)
        obs, reward, term, trunc, _ = env.step({'yaw': delta})
        r += reward
        done = term or trunc
    return r[0]


def train_pfns4bo(sim_data, nt, device, epochs=10):
    feature_cols = [f"yaw_{i}" for i in range(nt)] + ['freewind_measurements_0', 'freewind_measurements_1']
    num_features = len(feature_cols)

    X_all = torch.tensor(sim_data[feature_cols].values, dtype=torch.float32)
    y_raw = torch.tensor(sim_data[[f'power_{i}' for i in range(nt)]].sum(axis=1).values, dtype=torch.float32)
    y_mean = y_raw.mean()
    y_std = y_raw.std().clamp(min=1e-6)
    y_all = (y_raw - y_mean) / y_std

    def get_batch(batch_size, seq_len, num_features, device, hyperparameters, **kwargs):
        X_out = torch.zeros(seq_len, batch_size, num_features)
        Y_out = torch.zeros(seq_len, batch_size)
        for b in range(batch_size):
            idx = torch.randperm(len(X_all))[:seq_len]
            X_out[:, b, :] = X_all[idx]
            Y_out[:, b] = y_all[idx]
        return Batch(x=X_out.to(device), y=Y_out.to(device), target_y=Y_out.to(device))

    config = {
        'priordataloader_class': priors.get_batch_to_dataloader(get_batch),
        'encoder_generator': encoders.get_normalized_uniform_encoder(
            encoders.get_variable_num_features_encoder(encoders.Linear)
        ),
        'y_encoder_generator': encoders.Linear,
        'emsize': 512,
        'nhead': 4,
        'nhid': 1024,
        'nlayers': 6,
        'epochs': epochs,
        'lr': 0.0001,
        'bptt': 60,
        'batch_size': 128,
        'steps_per_epoch': 512,
        'warmup_epochs': 2,
        'scheduler': utils.get_cosine_schedule_with_warmup,
        'aggregate_k_gradients': 2,
        'weight_decay': 0.0,
        'train_mixed_precision': False,
        'efficient_eval_masking': True,
        'single_eval_pos_gen': utils.get_uniform_single_eval_pos_sampler(50, min_len=1),
        'extra_prior_kwargs_dict': {'num_features': num_features, 'hyperparameters': {}},
        'verbose': False,
    }

    def get_ys(config, device):
        b = config['priordataloader_class'].get_batch_method(
            128, n_samples, num_features, epoch=0, device=device,
            hyperparameters={**config['extra_prior_kwargs_dict']['hyperparameters'],
                             'num_hyperparameter_samples_per_batch': -1}
        )
        return b.target_y.flatten()

    def add_criterion(config, device):
        n_buckets = max(10, min(100, len(y_all) // 5))
        return {**config, 'criterion': bar_distribution.FullSupportBarDistribution(
            bar_distribution.get_bucket_limits(n_buckets, ys=get_ys(config, device).cpu())
        )}

    result = pfns_train.train(**add_criterion(config, device))
    model = result[2]
    return model, y_mean, y_std


def eval_pfns4bo(env, sim_data, seed, device):
    nt = env.num_turbines
    feature_cols = [f"yaw_{i}" for i in range(nt)] + ['freewind_measurements_0', 'freewind_measurements_1']

    pfn_model, y_mean, y_std = train_pfns4bo(sim_data, nt, device)
    pfn_model = pfn_model.to(device)
    pfn_model.eval()

    X_all = torch.tensor(sim_data[feature_cols].values, dtype=torch.float32)
    y_raw = torch.tensor(sim_data[[f'power_{i}' for i in range(nt)]].sum(axis=1).values, dtype=torch.float32)
    y_norm = ((y_raw - y_mean) / y_std).unsqueeze(1)  # (N, 1)
    X_ctx = X_all.unsqueeze(1).to(device)              # (N, 1, n_features)
    y_ctx = y_ctx = ((y_raw - y_mean) / y_std).unsqueeze(1).to(device)  # (N, 1)

    obs = env.reset(seed=seed, options=OPTIONS)
    freewind = torch.tensor(obs['freewind_measurements'], dtype=torch.float32).to(device)

    yaws = torch.zeros(nt, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([yaws], lr=1.0)

    for _ in range(100):
        opt.zero_grad()
        x_query = torch.cat([yaws, freewind]).reshape(1, 1, -1)
        x_full = torch.cat([X_ctx, x_query], dim=0)
        y_full = torch.cat([y_ctx, torch.zeros(1, 1, device=device)], dim=0)
        x_full = torch.cat([X_ctx, x_query], dim=0)           # (N+1, 1, 5)
        y_full = torch.cat([y_ctx, torch.zeros(1, 1, device=device)], dim=0)  # (N+1, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            logits = pfn_model((x_full, y_full), single_eval_pos=len(X_ctx), only_return_standard_out=False)['standard']
        (-pfn_model.criterion.mean(logits)).backward()
        opt.step()
        yaws.data.clamp_(-40, 40)

    return ramp_and_eval(env, yaws.detach().cpu().numpy(), seed)


def eval_pfn(env, sim_data, seed):
    nt = env.num_turbines
    feature_cols = [f"yaw_{i}" for i in range(nt)] + ['freewind_measurements_0', 'freewind_measurements_1']
    X = sim_data[feature_cols]
    y = sim_data[[f'power_{i}' for i in range(nt)]].sum(axis=1)

    reg = TabPFNRegressor(random_state=seed, device=args.device)
    reg.fit(X, y)

    obs = env.reset(seed=seed, options=OPTIONS)
    freewind = obs['freewind_measurements']

    def objective(yaws):
        X_in = np.concatenate([yaws, freewind]).reshape(1, -1)
        return -reg.predict(X_in)[0]

    result = minimize(objective, x0=np.zeros(nt), method='Nelder-Mead',
                      options={'maxiter': 50, 'xatol': 0.1, 'fatol': 1e-5})
    target_yaws = np.clip(result.x, -40, 40)
    return ramp_and_eval(env, target_yaws, seed)


def eval_gp(env, sim_data, seed):
    nt = env.num_turbines
    feature_cols = [f"yaw_{i}" for i in range(nt)] + ['freewind_measurements_0', 'freewind_measurements_1']
    d_features = len(feature_cols)
    n_samples = min(GP_MAX_SAMPLES, len(sim_data))
    X_base = sim_data[feature_cols].values[:n_samples]
    Y = sim_data[[f'power_{i}' for i in range(nt)]].values[:n_samples].sum(axis=1, keepdims=True)

    train_X = torch.tensor(X_base, dtype=torch.double).to(args.device)
    train_Y = torch.tensor(Y, dtype=torch.double).to(args.device)

    torch.manual_seed(seed)
    from botorch.models import SingleTaskGP
    gp_model = SingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(d=d_features),
        covar_module=ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=d_features)),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_model(mll)
    gp_model.eval()

    obs = env.reset(seed=seed, options=OPTIONS)
    freewind = torch.tensor(obs['freewind_measurements'], dtype=torch.double).to(args.device)

    yaws = torch.zeros(nt, dtype=torch.double, device=args.device, requires_grad=True)
    optimizer = torch.optim.Adam([yaws], lr=1.0)
    for _ in range(100):
        optimizer.zero_grad()
        test_X = torch.cat([yaws, freewind]).unsqueeze(0)
        pred_power = gp_model.posterior(test_X).mean
        (-pred_power).backward()
        optimizer.step()
        yaws.data.clamp_(-40, 40)

    target_yaws = yaws.detach().cpu().numpy()
    return ramp_and_eval(env, target_yaws, seed)


# ============================================================
# Run sample efficiency experiment
# ============================================================
RESULTS_CSV = 'sample_efficiency_results.csv'

if os.path.exists(RESULTS_CSV):
    df_existing = pd.read_csv(RESULTS_CSV)
    all_results = df_existing.to_dict('records')
    done = set(zip(df_existing['layout'], df_existing['n_samples'], df_existing['seed']))
    print(f"Loaded {len(all_results)} existing results from {RESULTS_CSV}")
else:
    all_results = []
    done = set()

for layout in LAYOUTS:
    print(f"\n{'='*50}")
    print(f"Layout: {layout}")
    print(f"{'='*50}")

    env = make_env(layout)
    dn = eval_do_nothing(env)
    print(f"  Do-nothing: {dn:.1f}")

    for n_samples in SAMPLE_BUDGETS:
        for seed in range(args.n_seeds):
            actual_seed = args.seed + seed
            layout_short = layout.replace('_Floris', '')

            if (layout_short, n_samples, actual_seed) in done:
                print(f"  n={n_samples}, seed={actual_seed}... SKIP (cached)")
                continue

            print(f"  n={n_samples}, seed={actual_seed}...", end=" ")

            sim_data = collect_sim_data(env, n_samples, actual_seed)

            pfn_score = eval_pfn(env, sim_data, actual_seed)
            pfns4bo_score = eval_pfns4bo(env, sim_data, actual_seed, args.device)

            if n_samples <= GP_MAX_SAMPLES:
                gp_score = eval_gp(env, sim_data, actual_seed)
            else:
                gp_score = np.nan

            print(f"TabPFN={pfn_score:.1f}, PFNs4BO={pfns4bo_score:.1f}, GP={gp_score:.1f}")

            all_results.append({
                'layout': layout_short,
                'n_samples': n_samples,
                'seed': actual_seed,
                'pfn': pfn_score,
                'pfns4bo': pfns4bo_score,
                'gp': gp_score,
                'do_nothing': dn,
            })

            pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

df = pd.DataFrame(all_results)
print('Finished computing...')

# ============================================================
# Plot sample efficiency curves
# ============================================================
base = {'Turb3_Row1': [239.5, 237.7], 'Ablaincourt': [351.0, 351.7]}

fig, axes = plt.subplots(1, len(LAYOUTS), figsize=(7 * len(LAYOUTS), 5))
if len(LAYOUTS) == 1:
    axes = [axes]

for ax, layout in zip(axes, LAYOUTS):
    layout_short = layout.replace('_Floris', '')
    sub = df[df['layout'] == layout_short]

    pfn_stats = sub.groupby('n_samples')['pfn'].agg(['mean', 'std']).reset_index()
    ax.plot(pfn_stats['n_samples'], pfn_stats['mean'], 'o-', label='TabPFN', color='#2196F3')
    ax.fill_between(pfn_stats['n_samples'],
                     pfn_stats['mean'] - pfn_stats['std'],
                     pfn_stats['mean'] + pfn_stats['std'],
                     alpha=0.2, color='#2196F3')

    pfns4bo_stats = sub.groupby('n_samples')['pfns4bo'].agg(['mean', 'std']).reset_index()
    ax.plot(pfns4bo_stats['n_samples'], pfns4bo_stats['mean'], 'o-', label='PFNs4BO', color='#9C27B0')
    ax.fill_between(pfns4bo_stats['n_samples'],
                     pfns4bo_stats['mean'] - pfns4bo_stats['std'],
                     pfns4bo_stats['mean'] + pfns4bo_stats['std'],
                     alpha=0.2, color='#9C27B0')

    gp_stats = sub.dropna(subset=['gp']).groupby('n_samples')['gp'].agg(['mean', 'std']).reset_index()
    if len(gp_stats) > 0:
        ax.plot(gp_stats['n_samples'], gp_stats['mean'], 's-', label='GP', color='#FF9800')
        ax.fill_between(gp_stats['n_samples'],
                         gp_stats['mean'] - gp_stats['std'],
                         gp_stats['mean'] + gp_stats['std'],
                         alpha=0.2, color='#FF9800')

    dn_val = sub['do_nothing'].iloc[0]
    ax.axhline(y=dn_val, color='gray', linestyle='--', label='Do-nothing', alpha=0.7)

    ax.axhline(y=base[layout_short][0], color='darkgreen', linestyle='--', label='IPPO_WFCRL', alpha=0.7)
    ax.axhline(y=base[layout_short][1], color='magenta', linestyle='--', label='MAPPO_WFCRL', alpha=0.7)

    ax.set_xscale('log')
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Episode Return (Sc. 1)')
    ax.set_title(layout_short)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_efficiency.png', bbox_inches='tight')

# ============================================================
# Yaw sweep: predicted total power vs yaw_0 (others at 0)
# ============================================================
SWEEP_SAMPLES = 500
SWEEP_SEED = args.seed

fig, axes = plt.subplots(1, len(LAYOUTS), figsize=(7 * len(LAYOUTS), 5))
if len(LAYOUTS) == 1:
    axes = [axes]

yaws_sweep = np.linspace(-40, 40, 100)

for ax, layout in zip(axes, LAYOUTS):
    layout_short = layout.replace('_Floris', '')
    env = make_env(layout)
    nt = env.num_turbines
    feature_cols = [f"yaw_{i}" for i in range(nt)] + ['freewind_measurements_0', 'freewind_measurements_1']
    d_features = len(feature_cols)
    d_total = d_features + 1

    sim_data = collect_sim_data(env, SWEEP_SAMPLES, SWEEP_SEED)
    X = sim_data[feature_cols]
    y_df = sim_data[[f'power_{i}' for i in range(nt)]]

    # --- TabPFN ---
    models = []
    for i in range(nt):
        reg = TabPFNRegressor(random_state=SWEEP_SEED, device=args.device)
        reg.fit(X, y_df[f'power_{i}'])
        models.append(reg)

    pfn_preds = []
    for y in yaws_sweep:
        X_in = np.zeros((1, d_features))
        X_in[0, 0] = y
        X_in[0, -2] = 8.0
        X_in[0, -1] = 270.0
        pfn_preds.append(sum(m.predict(X_in)[0] for m in models))
    ax.plot(yaws_sweep, pfn_preds, label='TabPFN', color='#2196F3')

    # --- PFNs4BO ---
    pfn_model, y_mean, y_std = train_pfns4bo(sim_data, nt, args.device)
    pfn_model.eval()
    X_ctx = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1).to(args.device)
    y_raw = torch.tensor(y_df.sum(axis=1).values, dtype=torch.float32)
    y_ctx = ((y_raw - y_mean) / y_std).unsqueeze(1).to(args.device)
    freewind_sweep = torch.tensor([8.0, 270.0], dtype=torch.float32).to(args.device)

    pfns4bo_preds = []
    with torch.no_grad():
        for y in yaws_sweep:
            yaw_vec = torch.zeros(nt, dtype=torch.float32, device=args.device)
            yaw_vec[0] = y
            x_query = torch.cat([yaw_vec, freewind_sweep]).reshape(1, 1, -1)
            x_full = torch.cat([X_ctx, x_query], dim=0)
            y_full = torch.cat([y_ctx, torch.zeros(1, 1, device=args.device)], dim=0)
            logits = pfn_model(x_full, y_full, single_eval_pos=len(X))
            pred = pfn_model.criterion.mean(logits).item() * y_std.item() + y_mean.item()
            pfns4bo_preds.append(pred)
    ax.plot(yaws_sweep, pfns4bo_preds, label='PFNs4BO', color='#9C27B0')

    # --- GP ---
    X_base = sim_data[feature_cols].values
    powers = y_df.values
    n = len(X_base)
    X_long = np.repeat(X_base, nt, axis=0)
    task_col = np.tile(list(range(nt)), n).reshape(-1, 1)
    X_long = np.hstack([X_long, task_col])
    Y_long = powers.reshape(-1, 1)

    train_X = torch.tensor(X_long, dtype=torch.double).to(args.device)
    train_Y = torch.tensor(Y_long, dtype=torch.double).to(args.device)

    torch.manual_seed(SWEEP_SEED)
    gp_model = MultiTaskGP(
        train_X, train_Y, task_feature=-1,
        input_transform=Normalize(d=d_total, indices=list(range(d_features))),
        covar_module=ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=d_features)),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_model(mll)
    gp_model.eval()

    gp_preds = []
    for y in yaws_sweep:
        X_in = torch.zeros(nt, d_total, dtype=torch.double).to(args.device)
        X_in[:, 0] = y
        X_in[:, nt] = 8.0
        X_in[:, nt + 1] = 270.0
        for t in range(nt):
            X_in[t, -1] = t
        gp_preds.append(gp_model.posterior(X_in).mean.sum().detach().cpu().numpy())
    ax.plot(yaws_sweep, gp_preds, label='GP', color='#FF9800')

    ax.set_xlabel('Yaw angle (turbine 0)')
    ax.set_ylabel('Predicted total power')
    ax.set_title(f'{layout_short} — Learned power vs yaw')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('yaw_sweep_comparison.png', bbox_inches='tight')