import secrets
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from tqdm import tqdm

from wfcrl import environments as envs
from kg_pfn import Transformer, AxialTransformer, WindFarmDataset, AxialWindFarmDataset, set_seed
from utils import step_policy, obs_to_row

import warnings
warnings.filterwarnings("ignore")

for var in ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK',
            'SLURM_PROCID', 'SLURM_NTASKS', 'SLURM_NODELIST']:
    os.environ.pop(var, None)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=27)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_steps', type=int, default=150)
parser.add_argument('--n_seeds', type=int, default=3)
args = parser.parse_args()

OPTIONS = {"wind_speed": 8, "wind_direction": 270}
LAYOUTS = [x for x in envs.list_envs() if 'Floris' in x and 'Dec' not in x and x != 'Turb1_Row1_Floris']

GP_SAMPLES = 500
PFN_SAMPLES = 100_000

N_BUCKETS = 50
D_MODEL = 1024
NHEAD = 8
FF_RATIO = 4
DROPOUT = 0.3
PFN_EPOCHS = 50
EARLY_STOP_PATIENCE = 10


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
    while len(rows) <= n_rows:
        run_id = secrets.token_urlsafe(6)
        obs = env.reset(seed=seed) # get sims with random conditions
        i, done = 0, False
        while not done:
            action = step_policy(i, env)
            obs, reward, term, trunc, info = env.step(action)
            rows.append(obs_to_row(obs, run_id, reward, info["power"], i))
            i += 1
            done = term or trunc
    return pd.DataFrame(rows[:n_rows])


def ramp_and_eval(env, target_yaws, seed):
    obs = env.reset(seed=seed, options=OPTIONS)
    r, done = 0, False
    while not done:
        delta = np.clip(target_yaws - obs['yaw'], -5, 5)
        obs, reward, term, trunc, _ = env.step({'yaw': delta})
        r += reward
        done = term or trunc
    return r[0]


def eval_do_nothing(env):
    env.reset(options=OPTIONS)
    r = 0
    for _ in range(args.max_steps):
        _, reward, _, _, _ = env.step({'yaw': np.zeros(env.num_turbines)})
        r += reward
    return r[0]


def eval_random(env, seed):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed, options=OPTIONS)
    r, done = 0, False
    while not done:
        _, reward, term, trunc, _ = env.step({'yaw': rng.uniform(-5, 5, env.num_turbines)})
        r += reward
        done = term or trunc
    return r[0]


def eval_gp(env, sim_data, seed):
    nt = env.num_turbines
    feature_cols = (
        [f"yaw_{i}" for i in range(nt)]
        + ['freewind_measurements_0', 'freewind_measurements_1']
        + [f"wind_speed_{i}" for i in range(nt)]
        + [f"wind_direction_{i}" for i in range(nt)]
        + [f"x_pos_{i}" for i in range(nt)]
        + [f"y_pos_{i}" for i in range(nt)]
    )
    X = torch.tensor(sim_data[feature_cols].values, dtype=torch.double).to(args.device)
    Y = torch.tensor(sim_data['reward'].values, dtype=torch.double).to(args.device).reshape(-1, 1)

    torch.manual_seed(seed)
    gp = SingleTaskGP(
        X, Y,
        input_transform=Normalize(d=len(feature_cols)),
        covar_module=ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=len(feature_cols))),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_model(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()

    obs = env.reset(seed=seed, options=OPTIONS)
    freewind = torch.tensor(obs['freewind_measurements'], dtype=torch.double).to(args.device)
    ws = torch.tensor(obs['wind_speed'], dtype=torch.double).to(args.device)
    wd = torch.tensor(obs['wind_direction'], dtype=torch.double).to(args.device)
    xc = torch.tensor(env.unwrapped.farm_case.xcoords, dtype=torch.double).to(args.device)
    yc = torch.tensor(env.unwrapped.farm_case.ycoords, dtype=torch.double).to(args.device)

    yaws = torch.zeros(nt, dtype=torch.double, device=args.device, requires_grad=True)
    opt = torch.optim.Adam([yaws], lr=1.0)
    for _ in range(100):
        opt.zero_grad()
        x = torch.cat([yaws, freewind, ws, wd, xc, yc]).unsqueeze(0)
        (-gp.posterior(x).mean).backward()
        opt.step()
        yaws.data.clamp_(-40, 40)

    return ramp_and_eval(env, yaws.detach().cpu().numpy(), seed)


def train_kg_pfn(sim_data, nt, seed, device):
    set_seed(seed)
    train_runs, val_runs = train_test_split(
        sim_data['run_id'].unique(), train_size=0.9, random_state=seed
    )
    train_ds = WindFarmDataset(sim_data[sim_data['run_id'].isin(train_runs)], num_turbines=nt)
    val_ds = WindFarmDataset(sim_data[sim_data['run_id'].isin(val_runs)], num_turbines=nt, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    input_channels = next(iter(train_loader))[0].shape[-1]
    y_flat = torch.cat([y for _, y in train_loader]).flatten()
    bucket_borders = torch.quantile(y_flat, torch.linspace(0, 1, N_BUCKETS + 1)).to(device)
    bucket_centers = (bucket_borders[:-1] + bucket_borders[1:]) / 2

    model = Transformer(
        input_channels=input_channels, n_buckets=N_BUCKETS,
        d_model=D_MODEL, nhead=NHEAD, ff_ratio=FF_RATIO,
        dropout=DROPOUT, device=device,
    )
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in tqdm(range(PFN_EPOCHS), desc="PFN training", leave=False):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            seq = x.shape[1]
            pos = torch.randint(1, seq, (1,)).item()
            mask = torch.zeros(seq, seq, device=device)
            mask[:, pos:] = float('-inf')
            opt.zero_grad()
            logits = model(x, mask, is_causal=False)
            loss = criterion(logits[:, pos, :], torch.bucketize(y, bucket_borders).clamp(0, N_BUCKETS - 1)[:, pos])
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                seq = x.shape[1]
                pos = torch.randint(1, seq, (1,)).item()
                mask = torch.zeros(seq, seq, device=device)
                mask[:, pos:] = float('-inf')
                logits = model(x, mask, is_causal=False)
                val_losses.append(criterion(logits[:, pos, :], torch.bucketize(y, bucket_borders).clamp(0, N_BUCKETS - 1)[:, pos]).item())

        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: v.cpu().clone() for name, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch}, best val loss: {best_val_loss:.4f}")
                break

    model.load_state_dict({name: v.to(device) for name, v in best_state.items()})
    return model, train_ds.scaler, bucket_centers, bucket_borders


def eval_kg_pfn(env, sim_data, seed, device):
    nt = env.num_turbines
    model, scaler, bucket_centers, bucket_borders = train_kg_pfn(sim_data, nt, seed, device)
    model.eval()

    feature_cols = (
        [f"yaw_{i}" for i in range(nt)]
        + ['freewind_measurements_0', 'freewind_measurements_1']
        + [f"wind_speed_{i}" for i in range(nt)]
        + [f"wind_direction_{i}" for i in range(nt)]
        + [f"x_pos_{i}" for i in range(nt)]
        + [f"y_pos_{i}" for i in range(nt)]
    )
    all_cols = feature_cols + ['reward']
    col_idx = {c: i for i, c in enumerate(all_cols)}

    wind_cols = (
        ['freewind_measurements_0', 'freewind_measurements_1']
        + [f"wind_speed_{i}" for i in range(nt)]
        + [f"wind_direction_{i}" for i in range(nt)]
        + [f"x_pos_{i}" for i in range(nt)]
        + [f"y_pos_{i}" for i in range(nt)]
    )
    wind_idx = [col_idx[c] for c in wind_cols]
    wind_mean = torch.tensor(scaler.mean_[wind_idx], dtype=torch.float32, device=device)
    wind_scale = torch.tensor(scaler.scale_[wind_idx], dtype=torch.float32, device=device)
    yaw_mean = torch.tensor(scaler.mean_[:nt], dtype=torch.float32, device=device)
    yaw_scale = torch.tensor(scaler.scale_[:nt], dtype=torch.float32, device=device)

    obs = env.reset(seed=seed, options=OPTIONS)
    xcoords = np.array(env.unwrapped.farm_case.xcoords)
    ycoords = np.array(env.unwrapped.farm_case.ycoords)
    freewind_raw = np.concatenate([
        obs['freewind_measurements'], obs['wind_speed'], obs['wind_direction'],
        xcoords, ycoords
    ])
    freewind_norm = (torch.tensor(freewind_raw, dtype=torch.float32, device=device) - wind_mean) / wind_scale

    yaws = torch.zeros(nt, dtype=torch.float32, device=device, requires_grad=True)
    lr = 0.01 if nt > 10 else 0.1
    opt = torch.optim.Adam([yaws], lr=lr)
    for _ in range(200):
        opt.zero_grad()
        yaws_norm = (yaws - yaw_mean) / yaw_scale
        x_query = torch.cat([yaws_norm, freewind_norm]).reshape(1, 1, -1)
        logits = model(x_query, is_causal=False)
        mean = (torch.softmax(logits[:, 0, :], dim=-1) * bucket_centers).sum()
        if torch.isnan(mean):
            break
        (-mean).backward()
        torch.nn.utils.clip_grad_norm_([yaws], 1.0)
        opt.step()
        yaws.data.clamp_(-40, 40)

    if torch.isnan(yaws).any():
        print("  Warning: yaw optimization diverged, using zeros")
        yaws = torch.zeros(nt, device=device)

    return ramp_and_eval(env, yaws.detach().cpu().numpy(), seed)


def train_axial_pfn(layout_short, seed, device, data_dir='../data'):
    set_seed(seed)

    train_ds = AxialWindFarmDataset(data_dir, exclude=layout_short)
    val_ds = AxialWindFarmDataset(data_dir, only=layout_short, scaler=train_ds.get_scaler())

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    sample_x, sample_y, sample_mask = next(iter(train_loader))
    input_channels = sample_x.shape[-1]

    y_flat = torch.cat([y for _, y, _ in train_loader]).flatten()
    bucket_borders = torch.quantile(y_flat, torch.linspace(0, 1, N_BUCKETS + 1)).to(device)
    bucket_centers = (bucket_borders[:-1] + bucket_borders[1:]) / 2

    model = AxialTransformer(
        input_channels=input_channels, n_buckets=N_BUCKETS,
        d_model=D_MODEL, nhead=NHEAD, ff_ratio=FF_RATIO,
        dropout=DROPOUT, device=device,
    )
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in tqdm(range(PFN_EPOCHS), desc="Axial PFN training", leave=False):
        model.train()
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            E = x.shape[1]
            pos = torch.randint(1, E, (1,)).item()
            ep_mask = torch.zeros(E, E, device=device)
            ep_mask[:, pos:] = float('-inf')
            turb_pad_mask = ~mask

            opt.zero_grad()
            logits = model(x, turbine_mask=turb_pad_mask, episode_mask=ep_mask)
            targets = torch.bucketize(y, bucket_borders).clamp(0, N_BUCKETS - 1)
            loss = criterion(logits[:, pos, :], targets[:, pos])
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                E = x.shape[1]
                pos = torch.randint(1, E, (1,)).item()
                ep_mask = torch.zeros(E, E, device=device)
                ep_mask[:, pos:] = float('-inf')
                turb_pad_mask = ~mask
                logits = model(x, turbine_mask=turb_pad_mask, episode_mask=ep_mask)
                targets = torch.bucketize(y, bucket_borders).clamp(0, N_BUCKETS - 1)
                val_losses.append(criterion(logits[:, pos, :], targets[:, pos]).item())

        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: v.cpu().clone() for name, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"  Axial early stop at epoch {epoch}, best val loss: {best_val_loss:.4f}")
                break

    model.load_state_dict({name: v.to(device) for name, v in best_state.items()})
    return model, train_ds.get_scaler(), bucket_centers, bucket_borders


def eval_axialpfn(env, layout_short, seed, device):
    nt = env.num_turbines
    model, scaler, bucket_centers, bucket_borders = train_axial_pfn(layout_short, seed, device)
    model.eval()

    feat_mean, feat_std = scaler
    feat_mean = feat_mean.to(device)
    feat_std = feat_std.to(device)

    obs = env.reset(seed=seed, options=OPTIONS)
    xcoords = np.array(env.unwrapped.farm_case.xcoords)
    ycoords = np.array(env.unwrapped.farm_case.ycoords)

    # feature order: [yaw, wind_speed, wind_dir, x_pos, y_pos, ctx_0, ctx_1]
    wind_raw = torch.zeros(nt, feat_mean.shape[0], device=device)
    for i in range(nt):
        wind_raw[i, 1] = obs['wind_speed'][i]
        wind_raw[i, 2] = obs['wind_direction'][i]
        wind_raw[i, 3] = xcoords[i]
        wind_raw[i, 4] = ycoords[i]
        wind_raw[i, 5] = obs['freewind_measurements'][0]
        wind_raw[i, 6] = obs['freewind_measurements'][1]
    wind_norm = (wind_raw - feat_mean) / feat_std

    yaws = torch.zeros(nt, dtype=torch.float32, device=device, requires_grad=True)
    lr = 0.01 if nt > 10 else 0.1
    opt = torch.optim.Adam([yaws], lr=lr)

    for _ in range(200):
        opt.zero_grad()
        yaws_norm = (yaws - feat_mean[0]) / feat_std[0]
        x_query = wind_norm.clone()
        x_query[:, 0] = yaws_norm
        # pad to max_turbines: (1, 1, max_T, F)
        x_padded = torch.zeros(1, 1, 100, feat_mean.shape[0], device=device)
        x_padded[0, 0, :nt, :] = x_query
        turb_pad_mask = torch.ones(1, 100, dtype=torch.bool, device=device)
        turb_pad_mask[0, :nt] = False  # True = padded
        logits = model(x_padded, turbine_mask=turb_pad_mask)
        mean = (torch.softmax(logits[:, 0, :], dim=-1) * bucket_centers).sum()
        if torch.isnan(mean):
            break
        (-mean).backward()
        torch.nn.utils.clip_grad_norm_([yaws], 1.0)
        opt.step()
        yaws.data.clamp_(-40, 40)

    if torch.isnan(yaws).any():
        print("  Warning: axial yaw optimization diverged, using zeros")
        yaws = torch.zeros(nt, device=device)

    return ramp_and_eval(env, yaws.detach().cpu().numpy(), seed)


# ============================================================
# Main loop
# ============================================================
os.makedirs(f'./experiment_3', exist_ok=True)
RESULTS_CSV = './experiment_3/results.csv'

if os.path.exists(RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)
    all_results = df.to_dict('records')
    done = set(zip(df['layout'], df['seed']))
    print(f"Loaded {len(all_results)} existing results from {RESULTS_CSV}")
else:
    all_results = []
    done = set()

for layout in LAYOUTS:
    layout_short = layout.replace('_Floris', '')

    if all((layout_short, args.seed + i) in done for i in range(args.n_seeds)):
        print(f"\nSkipping {layout_short} (all seeds cached)")
        continue

    print(f"\n{'='*50}\nLayout: {layout}\n{'='*50}")
    env = make_env(layout)

    dn = eval_do_nothing(env)
    print(f"  Do-nothing: {dn:.1f}")

    if not os.path.exists(f'../data/{layout_short}.csv'):
        pfn_data = collect_sim_data(env, PFN_SAMPLES, seed=27)
    else:
        pfn_data = pd.read_csv(f'../data/{layout_short}.csv')

    # inject coords if missing
    nt = env.num_turbines
    xcoords = np.array(env.unwrapped.farm_case.xcoords)
    ycoords = np.array(env.unwrapped.farm_case.ycoords)
    for i in range(nt):
        if f'x_pos_{i}' not in pfn_data.columns:
            pfn_data[f'x_pos_{i}'] = xcoords[i]
            pfn_data[f'y_pos_{i}'] = ycoords[i]

    gp_data = pfn_data[:GP_SAMPLES].copy()

    for seed_offset in range(args.n_seeds):
        seed = args.seed + seed_offset

        if (layout_short, seed) in done:
            print(f"  seed={seed}... SKIP (cached)")
            continue

        print(f"  seed={seed}...", end=" ", flush=True)

        rand_score = eval_random(env, seed)
        gp_score = eval_gp(env, gp_data, seed)
        pfn_score = eval_kg_pfn(env, pfn_data, seed, args.device)
        axialpfn_score = eval_axialpfn(env, layout_short, seed, args.device)
        
        print(f"Random={rand_score:.1f}, GP={gp_score:.1f}, KG-PFN={pfn_score:.1f}, AXIAL-PFN={axialpfn_score:.1f}")

        all_results.append({
            'layout': layout_short,
            'seed': seed,
            'do_nothing': dn,
            'random': rand_score,
            'gp': gp_score,
            'kg_pfn': pfn_score,
            'axialpfn': axialpfn_score,
        })
        pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

df = pd.DataFrame(all_results)
print('\nAll done.')

# ============================================================
# Bar chart
# ============================================================
layout_shorts = sorted(df['layout'].unique())

methods = ['random', 'gp', 'kg_pfn', 'axialpfn']
labels = ['Random', 'GP', 'PFN4BO', 'AXIAL-PFN']
colors = ['#FF9800', '#2196F3', '#9C27B0', '#E91E63']

x = np.arange(len(layout_shorts))
width = 0.2

fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(layout_shorts)), 5))
for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
    means, stds = [], []
    for ls in layout_shorts:
        sub = df[df['layout'] == ls]
        delta = sub[method] - sub['do_nothing']
        means.append(delta.mean())
        stds.append(delta.std())
    offset = (i - len(methods) / 2 + 0.5) * width
    ax.bar(x + offset, means, width, label=label, color=color, alpha=0.85)
    ax.errorbar(x + offset, means, yerr=stds, fmt='none', color='black', capsize=3)

ax.axhline(0, color='gray', linestyle='--', alpha=0.7, label='Do-Nothing')
ax.set_xticks(x)
ax.set_xticklabels(layout_shorts, rotation=45, ha='right')
ax.set_ylabel('Episode Return relative to Do-Nothing')
ax.set_title('Improvement over Do-Nothing by Layout')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./experiment_3/baseline_barchart.png', bbox_inches='tight')