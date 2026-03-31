import argparse
import secrets
import os
import random
import numpy as np
import pandas as pd
from wfcrl import environments as envs
from utils import step_policy, obs_to_row

parser = argparse.ArgumentParser()
parser.add_argument('--n_runs', type=int, default=1000)
parser.add_argument('--max_steps', type=int, default=150)
parser.add_argument('--seed', type=int, default=27)
parser.add_argument('--outdir', type=str, default='../data')
args = parser.parse_args()

LAYOUTS = [x for x in envs.list_envs() if 'Floris' in x and 'Dec' not in x and x != 'Turb1_Row1_Floris']

os.makedirs(args.outdir, exist_ok=True)
existing = [x.split('.')[0] for x in os.listdir(args.outdir)]

for layout in LAYOUTS:
    if layout in existing:
        print(f"Layout {layout} already exists")
        continue
        
    print(f"{layout}...", end=" ", flush=True)
    env = envs.make(
        layout,
        max_num_steps=args.max_steps,
        controls={"yaw": (-5, 5)},
        continuous_control=True,
        log=True,
    )

    nt = env.num_turbines
    xcoords = np.array(env.unwrapped.farm_case.xcoords)
    ycoords = np.array(env.unwrapped.farm_case.ycoords)

    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = []
    for ep in range(args.n_runs):
        run_id = secrets.token_urlsafe(6)
        obs = env.reset(seed=args.seed + ep)
        i, done = 0, False
        while not done:
            action = step_policy(i, env)
            obs, reward, term, trunc, info = env.step(action)
            row = obs_to_row(obs, run_id, reward, info["power"], i)
            for t in range(nt):
                row[f'x_pos_{t}'] = xcoords[t]
                row[f'y_pos_{t}'] = ycoords[t]
            rows.append(row)
            i += 1
            done = term or trunc

    df = pd.DataFrame(rows)
    fname = layout + '.csv'
    df.to_csv(os.path.join(args.outdir, fname), index=False)
    print(f"{len(df)} rows, {nt} turbines, {args.n_runs} episodes")

print("Done.")