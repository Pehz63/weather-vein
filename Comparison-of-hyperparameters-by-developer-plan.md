# Plan: Comprehensive Hyperparameter Comparison (All 3 Developers)

## Context

Zeph started a hyperparameter comparison document for his PPO implementation at `RL_PPO/RL_PPO_sc2_hyperparameter_comparison.md`, comparing his Scenario 2 PPO settings against the WFCRL reference paper (arXiv:2501.13592, Table 5). The task is to create a **single new markdown file** that captures all three developers' hyperparameters in a unified format, building on Zeph's structure.

The three developers and their work:
- **Kevin Babashov** (`origin/Kevin` branch, `CSCI_5980_notebooks/`): Surrogate-based methods — Do-Nothing, Random, GP (sklearn RBF), TabPFN (hosted), GraphPFN (GATv2Conv)
- **Zeph Johnson** (`origin/Zeph` branch, `RL_PPO/`): Deep RL — PPO with actor-critic, two scenarios
- **Aleksei Rozanov** (`origin/aleksei` branch, root notebooks): Surrogate-based — GP (BoTorch Matern) and TabPFN (local checkpoint)

---

## Output File

**Path:** `results/hyperparameter_comparison.md`

Rationale: `results/` already holds the cross-developer `testing_notes.md`; a hyperparameter comparison belongs there rather than under a single developer's folder.

---

## Document Structure

### 1. Preamble
One paragraph explaining this extends Zeph's detailed comparison at `RL_PPO/RL_PPO_sc2_hyperparameter_comparison.md` to cover all three developers.

### 2. Developer Overview Table
| Developer | Methods | Scenario coverage |
|---|---|---|
| **Kevin** | Do-Nothing, Random, GP (sklearn RBF), TabPFN (hosted), GraphPFN (GATv2) | Sc1, Sc2 |
| **Aleksei** | GP (BoTorch Matern), TabPFN (local), PFNs4BO, KG-PFN, Axial PFN | Sc1, Sc2, Sc3 |
| **Zeph** | PPO (actor-critic RL) | Sc1, Sc2 |

### 3. Shared / Environment Settings

| Parameter | Kevin | Aleksei | Zeph |
|---|---|---|---|
| Episode length | 150 | 150 | 150 |
| Seeds | [0, 1, 2] | 27 (experiments); 13 (notebooks) | 13 |
| Sc1 training wind | 8 m/s / 270° (fixed) | 8 m/s / 270° (fixed) | 8 m/s / 270° (fixed) |
| Sc1 test wind | 8 m/s / 270° | 8 m/s / 270° | 8 m/s / 270° |
| Sc2 training wind | Weibull(scale=8, shape=8) speed; N(270°, 20°) dir | random (no OPTIONS) | Weibull speed; N(270°, 20°) dir |
| Sc2 test wind | 8 m/s / 270° (fixed) | 8 m/s / 270° (fixed) | 8 m/s / 270° (fixed) |
| Sc3 training wind | — | multi-layout diverse (Axial PFN only) | — |
| Yaw optimization range | [-40°, 40°] | [-40°, 40°] (experiments); [-5°, 5°] (notebooks) | [-45°, 45°] |
| Baseline | do-nothing (all yaws = 0) | do-nothing (all yaws = 0) | no-control (zero delta) |

### 4. Kevin — Surrogate Methods

**4a. Sampling & search (per scenario)**
| Parameter | Scenario 1 | Scenario 2 |
|---|---|---|
| Context samples (N_INITIAL) | 10 | 32 |
| Candidates searched (N_CANDIDATES) | 2048 | 1024 |
| Environments | 20 FLORIS layouts | same |
| Seeds | [0, 1, 2] | same |

**4b. Method-specific architecture / settings**
| Parameter | GP | TabPFN | GraphPFN |
|---|---|---|---|
| Framework / model | sklearn GaussianProcessRegressor | tabpfn-client (hosted API) | custom GNNReg (GATv2Conv) |
| Kernel | RBF (length_scale=1.0, bounds 1e-2–1e3) | — | — |
| GP n_restarts_optimizer | 2 | — | — |
| GAT layers | — | — | 2 |
| GAT heads per layer | — | — | 2 |
| GAT hidden dim | — | — | 64 |
| Edge dim | — | — | 3 |
| GNN training samples | — | — | 3000 (2600 train / 400 val) |
| GNN optimizer | — | — | AdamW, lr=3e-4 |
| GNN loss | — | — | MSELoss, 15 epochs |
| GNN candidate batch | — | — | 128 |
| Yaw normalization scale | — | — | GRAPH_YAW_SCALE=30.0, WS_SCALE=15.0 |
| Device | CPU | CPU | CUDA if available, else CPU |

### 5. Aleksei — Surrogate & Transformer Methods

Aleksei has three scenario tiers and five methods. The notebooks (`GP4WFCRL.ipynb`, `PFNs4WFCRL.ipynb`, `PFNsBO_4WFCRL.ipynb`, `KGPFN_workflow.ipynb`) are prototypes; the canonical implementations live in `experiments/`.

**5a. Data collection (all methods)**
| Parameter | GP | TabPFN | PFNs4BO | KG-PFN | Axial PFN |
|---|---|---|---|---|---|
| Training samples | 500 | 500–1000 | 100k | 100k | multi-layout |
| Collection policy | 30% control prob, uniform[-5°, 5°] delta | same | same | same | same |

**5b. Model architecture**
| Parameter | GP | TabPFN | PFNs4BO | KG-PFN | Axial PFN |
|---|---|---|---|---|---|
| Kernel / base model | Matern(ν=0.5, ARD) in ScaleKernel | TabPFN v2 (local .ckpt) | Transformer | Transformer | Axial Transformer |
| BoTorch transforms | Normalize(d=features) + Standardize | — | — | — | — |
| Embedding dim (d_model / emsize) | — | — | 512 | 1024 | 128 |
| Transformer heads (nhead) | — | — | 4 | 8 | 8 |
| FF hidden (nhid / ff_ratio) | — | — | 1024 | 4× d_model | 4× d_model |
| Transformer layers (nlayers) | — | — | 6 | — | — |
| Output buckets (n_buckets) | — | — | 100 (dynamic) | 50 | 50 |
| Context length (bptt) | — | — | 150 | 150 | varies |
| Dropout | — | — | 0.0 | 0.3 | 0.3 |

**5c. Training settings**
| Parameter | GP | TabPFN | PFNs4BO | KG-PFN | Axial PFN |
|---|---|---|---|---|---|
| Optimizer | Adam lr=1.0, 100 iter | — | AdamW lr=1e-4 | AdamW lr=1e-4 | AdamW lr=1e-3 |
| Epochs | — | — | 1 (fast iter) | 50 | 25 |
| Batch size | — | — | 128 | 128 | 32 |
| Steps/epoch | — | — | 512 | — | — |
| Warmup epochs | — | — | 5 | — | — |
| Early stop patience | — | — | — | 10 | 10 |

**5d. Yaw optimization (at eval time)**
| Parameter | GP (notebook) | GP (experiments) | TabPFN | PFNs4BO / KG-PFN / Axial |
|---|---|---|---|---|
| Optimizer | Adam lr=1.0 | Adam lr=1.0 | Nelder-Mead | Adam lr=0.1 (≤10 turbines) |
| Iterations | 100 | 100 | maxiter=50, ftol=1e-5 | 200 |
| Yaw bounds | [-5°, 5°] | [-40°, 40°] | [-5°, 5°] (per step) | [-40°, 40°] |
| Gradient clipping | — | — | — | norm=1.0 |

**5e. Scenario differences (experiment scripts)**
| Aspect | Sc1 (exp1.py) | Sc2 (exp2.py) | Sc3 (exp3.py) |
|---|---|---|---|
| Training data wind | Fixed 8 m/s / 270° | Random (no OPTIONS) | Multi-layout diverse |
| Model | KG-PFN (d_model=1024) | KG-PFN (d_model=1024) | Axial PFN (d_model=128) |
| PFN epochs | 50 | 50 | 25 |
| Eval wind | Fixed 8 m/s / 270° | Fixed 8 m/s / 270° | Fixed 8 m/s / 270° |

### 6. Zeph — PPO (Actor-Critic RL)

Condensed from `RL_PPO/RL_PPO_sc2_hyperparameter_comparison.md`; see that file for full paper comparison.

| Parameter | Scenario 1 | Scenario 2 | Paper (Table 5) |
|---|---|---|---|
| Actor-critic hidden dim | 128 | 256 | 64 |
| N_ENVS (parallel envs) | 8 | 16 | — |
| ROLLOUTS_PER_UPDATE | 8 | 16 | — |
| N_EPISODES (update cycles) | 63 | 63 (+extended) | ~97 |
| LR schedule | 3e-4 (flat) | 3e-4 → 1e-5 (linear) | 3e-4 → 0 |
| N_EPOCHS (per rollout) | 10 | 20 | 10 |
| Minibatch size | 32 | 32 | 64 |
| GAMMA (β) | 0.99 | 0.99 | 0.99 |
| GAE_LAMBDA | 0.95 | 0.95 | 0.95 |
| CLIP_EPS | 0.2 | 0.2 | 0.2 |
| VF_COEF | 0.5 | 0.5 | 0.5 |
| ENT_COEF | 0.01 → 0.001 (linear) | same | not reported |
| MAX_GRAD | 0.5 | 0.5 | 0.5 |
| REWARD_SCALE | 20.0 | 20.0 | raw power |
| TERMINAL_SCALE | 5.0 | 5.0 | — |
| CONVERGENCE_COEF | 0.05 | 0.05 | — |
| Warm-start from Sc1 | — | Yes (LOAD_CHECKPOINT=True) | — |
| Approx total steps | ~151k | ~480k | 200k |

### 7. Notable Cross-Developer Differences

- **Approach**: Zeph is the only RL developer; Kevin and Aleksei both use surrogate models for one-shot yaw optimization.
- **GP kernel**: Kevin uses sklearn RBF; Aleksei uses BoTorch Matern(ν=0.5) with ARD — the latter is more principled for wind-farm geometry.
- **PFN scope**: Kevin uses TabPFN as a black-box regressor; Aleksei trains full Transformer PFNs (PFNs4BO, KG-PFN) on 100k simulated episodes and adds an Axial PFN for multi-layout transfer (Sc3).
- **Yaw optimization at eval**: Kevin uses random candidate search (1024–2048 samples); Aleksei uses gradient-based Adam (100–200 iterations); Zeph's PPO outputs yaw deltas directly as policy actions.
- **Data scale**: Kevin collects 10–32 online episodes as context; Aleksei pre-trains on 100k offline simulations; Zeph trains 151k–480k RL environment steps.
- **Scenario 3**: Only Aleksei implements a third scenario (multi-layout transfer via Axial PFN with leave-one-out cross-validation).
- **Reward signal**: Zeph adds shaped differential reward (REWARD_SCALE=20, TERMINAL_SCALE=5, CONVERGENCE_COEF=0.05); Kevin and Aleksei optimize raw surrogate predictions of farm power.

---

## Source Files

| File | Branch | Data used |
|------|--------|-----------|
| `CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_Scenario1.ipynb` | Kevin | Sc1: N_INITIAL=10, N_CANDIDATES=2048, GP/TabPFN/GNN params |
| `CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_V2.ipynb` | Kevin | Sc2: N_INITIAL=32, N_CANDIDATES=1024, GRAPH_YAW/WS_SCALE |
| `GP4WFCRL.ipynb` | aleksei | GP notebook: Matern kernel, BoTorch MultiTaskGP, Adam 100 iter |
| `PFNs4WFCRL.ipynb` | aleksei | TabPFN notebook: v2 local ckpt, Nelder-Mead |
| `PFNsBO_4WFCRL.ipynb` | aleksei | PFNs4BO prototype: emsize=512, nhead=4, nlayers=6 |
| `KGPFN_workflow.ipynb` | aleksei | KG-PFN prototype: d_model=2056, nhead=8 |
| `experiments/experiment_1.py` | aleksei | Sc1 canonical: GP_SAMPLES=500, PFN_SAMPLES=100k, d_model=1024 |
| `experiments/experiment_2.py` | aleksei | Sc2 canonical: random wind training, same model as exp1 |
| `experiments/experiment_3.py` | aleksei | Sc3: Axial PFN d_model=128, epochs=25, AdamW lr=1e-3, multi-layout |
| `RL_PPO/RL_PPO_Scenario_1_full.ipynb` | Zeph | Sc1: hidden_dim=128, N_ENVS=8, reward shaping params |
| `RL_PPO/RL_PPO_Scenario_2_full.ipynb` | Zeph | Sc2: hidden_dim=256, N_ENVS=16, MIN_LR=1e-5, LOAD_CHECKPOINT |
| `RL_PPO/RL_PPO_sc2_hyperparameter_comparison.md` | Zeph | Full PPO vs paper comparison (referenced, not duplicated) |

---

## Verification

1. Confirm the file renders correctly (headers, table alignment)
2. Spot-check 3 values against source notebooks
3. Commit to `claude/collect-branch-results-U8GdI` and push

---

# Plan: Write Developer Testing Notes to Markdown File

## Context

The previous task collected results from Kevin, Zeph, and aleksei branches. The new request is to write a `.md` file with a table summarizing how each developer tested their models — with a specific follow-up question about what `wind_OPTS` were used for Scenario 1 and Scenario 2, and whether this differed across developers.

---

## Key Findings (from notebook exploration)

### Wind conditions at final evaluation — **this DOES differ by developer**

| Developer | Scenario 1 test wind | Scenario 2 test wind | Notes |
|-----------|---------------------|---------------------|-------|
| **Kevin** | `wind_speed=8, wind_direction=270` (fixed) | `wind_speed=8, wind_direction=270` (fixed) | Sc2 training uses random wind (Weibull/Normal), but `SCENARIO2_TEST_OPTIONS = SCENARIO1_OPTIONS.copy()` — test is still fixed |
| **Zeph** | `wind_speed=8, wind_direction=270` (fixed) | `WIND_OPTS = None` (random per episode) | Sc2 evaluation uses the same random distribution as training |
| **aleksei** | `wind_speed=8, wind_direction=270` (fixed) | — (not implemented) | Only Sc1 tested; notebooks mention Sc2 but no code executed it |

**Critical divergence:** Kevin and Zeph interpret "Scenario 2" differently at evaluation time. Kevin always tests at fixed 8 m/s / 270°; Zeph evaluates under random wind — so their Scenario 2 numbers are not directly comparable.

---

## File to Create

**Path:** `results/testing_notes.md`

---

## Content Plan

### Table 1: High-level testing methodology per developer

Columns: Developer | Methods | Simulator | Environments | Seeds/Runs | Evaluation Metric | Scenarios Tested

### Table 2: Wind conditions (wind_OPTS) by developer and scenario

Columns: Developer | Scenario 1 training wind | Scenario 1 test wind | Scenario 2 training wind | Scenario 2 test wind

### Table 3: Method-specific details

Columns: Developer | Method | Training data / episodes | Key hyperparameters | Baseline compared against

### Notes section

- Caveat that Kevin Sc2 and Zeph Sc2 are not directly comparable (different test wind protocols)
- Aleksei only completed Sc1; Sc2 noted as future work in notebooks
- Absolute reward values differ because Zeph evaluates under variable wind (higher variance)

---

## Critical Files (source of truth)

| Notebook | Branch | Key cells |
|----------|--------|-----------|
| `CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_V2.ipynb` | Kevin | Cell 16: `FIXED_FLORIS_TEST_OPTIONS`; Cell 27: `SCENARIO2_TEST_OPTIONS = SCENARIO1_OPTIONS.copy()` |
| `RL_PPO/RL_PPO_Scenario_1_full.ipynb` | Zeph | Cell 6: `WIND_OPTS = {"wind_speed": 8, "wind_direction": 270}` |
| `RL_PPO/RL_PPO_Scenario_2_full.ipynb` | Zeph | Cell 6: `WIND_OPTS = None` |
| `PFNs4WFCRL.ipynb` | aleksei | Cell 5: `options={"wind_speed": 8, "wind_direction": 270}` throughout |

---

## Verification

1. File created at `results/testing_notes.md`
2. Commit + push to `claude/collect-branch-results-U8GdI`

---

# Plan: Collect Branch Results into CSV + Bar Chart (previous task — COMPLETE)

## Context

Three contributor branches exist on `pehz63/weather-vein`:
- **Kevin** — PFN-based methods (GP, TabPFN, GraphPFN) vs baselines, FLORIS simulator, 20 layouts × 2 scenarios
- **Zeph** — PPO RL agent across 12 FLORIS environments, 2 scenarios
- **aleksei** — earlier PFN implementation, Ablaincourt only

The goal is to extract performance numbers from all three, unify into a CSV, and produce a bar chart of **gain% vs Do-Nothing baseline** for representative environment IDs.

Kevin pointed to these as the starting point (both in `CSCI_5980_notebooks/`):
- `WFCRL_GraphPFN_TabPFN_V2.ipynb` — full Sc1/Sc2 comparison
- `WFCRL_PFNs_FastFarm.ipynb` — FastFarm variant

---

## What Data Lives Where (confirmed from executed notebook outputs)

### Kevin branch — `CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_V2.ipynb`
- **Scenario 1** (train = test wind): Cell 22 stream outputs → per-seed rows `(layout, seed, method, reward)`; Cell 24 execute_result → 100-row aggregated table `(layout, method, mean_gain, std_gain, mean_reward, std_reward)`
- **Scenario 2** (train ≠ test wind): Cell 29 stream outputs → same per-seed format; Cell 30 → aggregated
- Methods: `Do-Nothing`, `Random`, `GP`, `TabPFN`, `GraphPFN`; 20 layouts × 3 seeds × 2 scenarios
- Notebook saves `results/scenario2_floris_graph_tab_pfn_summary.csv` locally (Colab only, not in git) → re-extract from cell outputs

### Zeph branch — `RL_PPO/RL_PPO_Scenario_2_full.ipynb` (and Scenario_1_full)
- **Cell 49** `execute_result` → clean summary DataFrame across 12 environments:
  `(env_id, status, n_turbines, ppo_mean, ppo_std, baseline_mean, baseline_std, gain_pct, simulator)`
- Same table appears in both Sc1 and Sc2 notebooks (Sc2 loads existing checkpoints + re-evaluates)
- Environments include: Turb1-32, Ablaincourt, HornsRev1, HornsRev2, Ormonde, WMR

### aleksei branch — `PFNs4WFCRL.ipynb`
- Ablaincourt_Floris only; Cell 11 stream: `Total reward = 351.2` (do-nothing FLORIS); Cell 14 stream: `Do-nothing total: [353.5]`
- Not directly comparable to Kevin's framework — include as annotation only

---

## Implementation

### File to create: `collect_results.py` (repo root)
Committed to `claude/collect-branch-results-U8GdI`.

**Step 1 — Fetch branches** (already done; `origin/Kevin`, `origin/Zeph`, `origin/aleksei` exist)

**Step 2 — Extract Kevin's V2 notebook**
```python
nb = json.loads(subprocess.check_output(
    ['git', 'show', 'origin/Kevin:CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_V2.ipynb']))
```
Parse Cell 24 (Sc1 aggregate) and Cell 30 (Sc2 aggregate) `execute_result` outputs using `pd.read_fwf(io.StringIO(text))`. Tag `branch='Kevin'`.

**Step 3 — Extract Zeph's RL_PPO summary**
```python
for scenario, nb_path in [('Scenario 1', 'RL_PPO/RL_PPO_Scenario_1_full.ipynb'),
                           ('Scenario 2', 'RL_PPO/RL_PPO_Scenario_2_full.ipynb')]:
    nb = json.loads(subprocess.check_output(['git', 'show', f'origin/Zeph:{nb_path}']))
```
Find Cell 49 `execute_result` output → parse as DataFrame. Compute `mean_gain = (ppo_mean - baseline_mean)` in kW; `gain_pct` already present. Tag `branch='Zeph'`, `method='PPO'`.

**Step 4 — Save unified CSV**
Schema: `branch, scenario, layout, method, mean_gain_kw, gain_pct, mean_reward, std_reward`
- Kevin: read directly from parsed aggregated tables (columns already match)
- Zeph: `mean_gain_kw = ppo_mean - baseline_mean`, rename `env_id` → `layout`
- Save to `results/combined_results.csv`

**Step 5 — Generate bar chart**
Representative environments (overlap between Kevin and Zeph):
```python
REP_ENVS = ['Ablaincourt_Floris', 'HornsRev1_Floris', 'Turb3_Row1_Floris', 'Turb16_Row5_Floris']
```
Two-panel figure (Scenario 1 top, Scenario 2 bottom):
- X-axis: 4 representative environments
- Grouped bars per env: `GP`, `TabPFN`, `GraphPFN` (Kevin), `PPO` (Zeph)
- Y-axis: `gain_pct` (% over Do-Nothing baseline)
- Error bars: `std_gain / do_nothing_mean * 100` for Kevin; `ppo_std / baseline_mean * 100` for Zeph
- Save to `results/performance_bar_chart.png`

---

## Critical Files

| File | Branch | Role |
|------|--------|------|
| `CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_V2.ipynb` | Kevin | Cell 24 (Sc1 agg), Cell 30 (Sc2 agg) |
| `RL_PPO/RL_PPO_Scenario_1_full.ipynb` | Zeph | Cell 49 PPO summary |
| `RL_PPO/RL_PPO_Scenario_2_full.ipynb` | Zeph | Cell 49 PPO summary |
| `PFNs4WFCRL.ipynb` | aleksei | Supplementary note only |
| `collect_results.py` | claude branch | Script to create |
| `results/combined_results.csv` | claude branch | Output |
| `results/performance_bar_chart.png` | claude branch | Output |

---

## Parsing Notes

Kevin Cell 24/30 outputs look like (fixed-width):
```
                layout      method  mean_gain  std_gain  mean_reward  std_reward
0   Ablaincourt_Floris  Do-Nothing   0.000000  0.000000   351.166729    0.000000
2   Ablaincourt_Floris          GP  13.610683  0.684547   364.777412    0.684547
```
Parse with `pd.read_fwf(io.StringIO(text), index_col=0)`.

Zeph Cell 49 looks like a standard DataFrame repr — same parsing approach.

---

## Verification

1. `python collect_results.py` → prints row counts per branch/scenario
2. `results/combined_results.csv` — expect ~100 rows from Kevin Sc1, ~100 Sc2, ~12 from Zeph Sc1, ~12 Sc2
3. `results/performance_bar_chart.png` — 2-panel, 4 env groups, 4 method bars each
4. Commit + push to `claude/collect-branch-results-U8GdI`