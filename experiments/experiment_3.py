import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from wfcrl import environments as envs
from kg_pfn import AxialTransformer, AxialWindFarmDataset, set_seed

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

N_BUCKETS = 50
D_MODEL = 128
NHEAD = 8
FF_RATIO = 4
DROPOUT = 0.3
PFN_EPOCHS = 25
EARLY_STOP_PATIENCE = 10


def make_env(layout):
    return envs.make(
        layout,
        max_num_steps=args.max_steps,
        controls={"yaw": (-5, 5)},
        continuous_control=True,
        log=True,
    )


def ramp_and_eval(env, target_yaws, seed):
    obs = env.reset(seed=seed, options=OPTIONS)
    r, done = 0, False
    while not done:
        delta = np.clip(target_yaws - obs['yaw'], -5, 5)
        obs, reward, term, trunc, _ = env.step({'yaw': delta})
        r += reward
        done = term or trunc
    return r[0]


def train_axial_pfn(layout_short, seed, device, data_dir='../data'):
    set_seed(seed)

    train_ds = AxialWindFarmDataset(data_dir, exclude=layout_short)
    val_ds = AxialWindFarmDataset(data_dir, only=layout_short, scaler=train_ds.get_scaler())

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    sample_x, sample_y, sample_mask = next(iter(train_loader))
    input_channels = sample_x.shape[-1]

    y_train = torch.cat([y for _, y, _ in train_loader]).flatten()
    y_val = torch.cat([y for _, y, _ in val_loader]).flatten()
    y_all = torch.cat([y_train, y_val])
    bucket_borders = torch.quantile(y_all, torch.linspace(0, 1, N_BUCKETS + 1)).to(device)
    bucket_centers = (bucket_borders[:-1] + bucket_borders[1:]) / 2

    model = AxialTransformer(
        input_channels=input_channels, n_buckets=N_BUCKETS,
        d_model=D_MODEL, nhead=NHEAD, ff_ratio=FF_RATIO,
        dropout=DROPOUT, device=device,
    )
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3)

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

            opt.zero_grad()
            logits = model(x, episode_mask=ep_mask)
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
                logits = model(x, episode_mask=ep_mask)
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

    if best_state is None:
        print("  Warning: no improvement during training, using final state")
        best_state = {name: v.cpu().clone() for name, v in model.state_dict().items()}
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
        x_query = torch.cat([
            yaws_norm.unsqueeze(1),
            wind_norm[:, 1:].detach()
        ], dim=1)
        x_input = x_query.unsqueeze(0).unsqueeze(0)  # (1, 1, nt, F)
        logits = model(x_input)
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
# Bootstrap from experiment 2
# ============================================================
os.makedirs('./experiment_3', exist_ok=True)
RESULTS_CSV = './experiment_3/results.csv'
EXP2_CSV = './experiment_2/results.csv'

if os.path.exists(RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)
elif os.path.exists(EXP2_CSV):
    df = pd.read_csv(EXP2_CSV)
    df['axialpfn'] = np.nan
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Bootstrapped from experiment 2: {len(df)} rows")
else:
    raise FileNotFoundError(f"Need either {RESULTS_CSV} or {EXP2_CSV} to exist")

# ============================================================
# Run only axial PFN for missing entries
# ============================================================
for layout in LAYOUTS:
    layout_short = layout.replace('_Floris', '')
    seeds = [args.seed + i for i in range(args.n_seeds)]

    sub = df[df['layout'] == layout_short]
    missing = [s for s in seeds if sub[sub['seed'] == s]['axialpfn'].isna().any() or s not in sub['seed'].values]

    if not missing:
        print(f"Skipping {layout_short} (axial PFN cached)")
        continue

    print(f"\n{'='*50}\nLayout: {layout}\n{'='*50}")
    env = make_env(layout)

    for seed in missing:
        print(f"  seed={seed}...", end=" ", flush=True)
        score = eval_axialpfn(env, layout_short, seed, args.device)
        print(f"AXIAL-PFN={score:.1f}")

        mask = (df['layout'] == layout_short) & (df['seed'] == seed)
        if mask.any():
            df.loc[mask, 'axialpfn'] = score
        else:
            row = {'layout': layout_short, 'seed': seed, 'axialpfn': score}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(RESULTS_CSV, index=False)

print('\nAll done.')

# ============================================================
# Bar chart
# ============================================================
df = pd.read_csv(RESULTS_CSV)
layout_shorts = sorted(df['layout'].unique())

methods = ['random', 'gp', 'kg_pfn', 'axialpfn']
labels = ['Random', 'GP', 'PFN4BO', 'Axial PFN']
colors = ['#FF9800', '#2196F3', '#9C27B0', '#E91E63']

x = np.arange(len(layout_shorts))
width = 0.2

fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(layout_shorts)), 5))
for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
    means, stds = [], []
    for ls in layout_shorts:
        sub = df[df['layout'] == ls].dropna(subset=[method])
        delta = sub[method] - sub['do_nothing']
        means.append(delta.mean() if len(delta) > 0 else 0)
        stds.append(delta.std() if len(delta) > 1 else 0)
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