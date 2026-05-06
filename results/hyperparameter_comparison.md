# Hyperparameter Comparison: All Developers

This document extends Zeph's detailed single-developer comparison in
`RL_PPO/RL_PPO_sc2_hyperparameter_comparison.md` to cover all three contributors —
Kevin Babashov, Aleksei Rozanov, and Zeph Johnson — in a unified format.

---

## Developer Overview

| Developer | Methods | Scenario coverage |
|---|---|---|
| **Kevin** | Do-Nothing, Random, GP (sklearn RBF), TabPFN (hosted API), GraphPFN (GATv2Conv) | Sc1, Sc2 |
| **Aleksei** | GP (BoTorch Matern), TabPFN (local), PFNs4BO, KG-PFN, Axial PFN | Sc1, Sc2, Sc3 |
| **Zeph** | PPO (actor-critic deep RL) | Sc1, Sc2 |

---

## Shared / Environment Settings

| Parameter | Kevin | Aleksei | Zeph |
|---|---|---|---|
| Episode length (max\_num\_steps) | 150 | 150 | 150 |
| Seeds | [0, 1, 2] | 27 (experiments); 13 (notebooks) | 13 |
| Sc1 training wind | 8 m/s / 270° (fixed) | 8 m/s / 270° (fixed) | 8 m/s / 270° (fixed) |
| Sc1 test wind | 8 m/s / 270° | 8 m/s / 270° | 8 m/s / 270° |
| Sc2 training wind | Weibull(scale=8, shape=8) speed; N(270°, 20°) dir | random (no OPTIONS set) | Weibull speed; N(270°, 20°) dir |
| Sc2 test wind | 8 m/s / 270° (fixed) | 8 m/s / 270° (fixed) | 8 m/s / 270° (fixed) |
| Sc3 training wind | — | multi-layout diverse (Axial PFN only) | — |
| Yaw optimization range | [-40°, 40°] | [-40°, 40°] (experiments); [-5°, 5°] (notebooks) | [-45°, 45°] |
| Baseline | do-nothing (all yaws = 0) | do-nothing (all yaws = 0) | no-control (zero yaw delta) |
| Simulator | FLORIS (via wfcrl) | FLORIS (via wfcrl) | FLORIS (via wfcrl) |

---

## Kevin — Surrogate Methods

Kevin evaluates five methods across up to 20 FLORIS layouts using a collect-then-optimize
loop: gather N\_INITIAL context episodes, then score N\_CANDIDATES random yaw configurations
through the surrogate and apply the best.

### Sampling and search (per scenario)

| Parameter | Scenario 1 | Scenario 2 |
|---|---|---|
| Context samples (N\_INITIAL) | 10 | 32 |
| Candidates searched (N\_CANDIDATES) | 2048 | 1024 |
| Environments evaluated | 20 FLORIS layouts | same |
| Seeds | [0, 1, 2] | same |

### Method-specific architecture / settings

| Parameter | GP | TabPFN | GraphPFN |
|---|---|---|---|
| Framework / model | sklearn `GaussianProcessRegressor` | tabpfn-client (hosted API) | custom `GNNReg` (GATv2Conv) |
| Kernel | RBF (length\_scale=1.0, bounds 1e-2–1e3) | — | — |
| GP n\_restarts\_optimizer | 2 | — | — |
| normalize\_y | True | — | — |
| GAT layers | — | — | 2 |
| GAT heads per layer | — | — | 2 |
| GAT hidden dim | — | — | 64 |
| Input / edge dim | 5 in, 3 edge | — | 5 in, 3 edge |
| GNN training samples | — | — | 3000 (2600 train / 400 val) |
| GNN optimizer | — | — | AdamW, lr=3e-4 |
| GNN loss | — | — | MSELoss, 15 epochs |
| GNN candidate batch size | — | — | 128 |
| Yaw normalization (GRAPH\_YAW\_SCALE) | — | — | 30.0 |
| Wind speed normalization (GRAPH\_WS\_SCALE) | — | — | 15.0 |
| Device | CPU | CPU | CUDA if available, else CPU |

---

## Aleksei — Surrogate and Transformer Methods

Aleksei implements five methods across three scenario tiers. The Jupyter notebooks
(`GP4WFCRL.ipynb`, `PFNs4WFCRL.ipynb`, `PFNsBO_4WFCRL.ipynb`, `KGPFN_workflow.ipynb`)
are prototyping environments; canonical, reproducible runs live in `experiments/`.

### Data collection (all methods)

| Parameter | GP | TabPFN | PFNs4BO | KG-PFN | Axial PFN |
|---|---|---|---|---|---|
| Training samples | 500 | 500–1000 | 100 000 | 100 000 | multi-layout (varies) |
| Collection policy | 30% control prob, uniform[-5°, 5°] per-step delta | same | same | same | same |

### Model architecture

| Parameter | GP | TabPFN | PFNs4BO | KG-PFN | Axial PFN |
|---|---|---|---|---|---|
| Kernel / base model | Matern(ν=0.5, ARD) in ScaleKernel | TabPFN v2 (local .ckpt) | Transformer | Transformer | Axial Transformer |
| BoTorch input / output transforms | Normalize + Standardize | — | — | — | — |
| Embedding dim (emsize / d\_model) | — | — | 512 | 1024 | 128 |
| Transformer heads (nhead) | — | — | 4 | 8 | 8 |
| FF hidden (nhid / ff\_ratio×d\_model) | — | — | 1024 | 4 096 | 512 |
| Transformer layers (nlayers) | — | — | 6 | — | — |
| Output buckets (n\_buckets) | — | — | 100 (dynamic) | 50 | 50 |
| Context length (bptt) | — | — | 150 | 150 | varies |
| Dropout | — | — | 0.0 | 0.3 | 0.3 |

### Training settings

| Parameter | GP | TabPFN | PFNs4BO | KG-PFN | Axial PFN |
|---|---|---|---|---|---|
| Optimizer | Adam lr=1.0, 100 iter | — | AdamW lr=1e-4 | AdamW lr=1e-4 | AdamW lr=1e-3 |
| Epochs | 100 iter (no epochs) | — | 1 (fast iteration) | 50 | 25 |
| Batch size | — | — | 128 | 128 | 32 |
| Steps per epoch | — | — | 512 | — | — |
| Warmup epochs | — | — | 5 | — | — |
| Early stop patience | — | — | — | 10 | 10 |

### Yaw optimization at evaluation time

| Parameter | GP (notebook) | GP (experiments) | TabPFN | PFNs4BO / KG-PFN / Axial |
|---|---|---|---|---|
| Optimizer | Adam lr=1.0 | Adam lr=1.0 | Nelder-Mead | Adam lr=0.1 (≤10 turbines) |
| Iterations / max iter | 100 | 100 | maxiter=50, ftol=1e-5 | 200 |
| Yaw bounds | [-5°, 5°] | [-40°, 40°] | [-5°, 5°] per step | [-40°, 40°] |
| Gradient clipping | — | — | — | norm=1.0 |

### Scenario differences (experiment scripts)

| Aspect | Sc1 (experiment\_1.py) | Sc2 (experiment\_2.py) | Sc3 (experiment\_3.py) |
|---|---|---|---|
| Training data wind | Fixed 8 m/s / 270° | Random (no OPTIONS) | Multi-layout diverse |
| Primary model | KG-PFN | KG-PFN | Axial PFN |
| d\_model | 1024 | 1024 | 128 |
| PFN training epochs | 50 | 50 | 25 |
| Axial PFN leave-one-out CV | — | — | Yes |
| Evaluation wind | Fixed 8 m/s / 270° | Fixed 8 m/s / 270° | Fixed 8 m/s / 270° |
| SLURM GPU allocation | a100 × 2, 24 h | a100 × 2, 24 h | a100 × 1, 12 h |

---

## Zeph — PPO (Actor-Critic Deep RL)

Condensed from `RL_PPO/RL_PPO_sc2_hyperparameter_comparison.md`; see that file for
the full comparison against the WFCRL reference paper (arXiv:2501.13592, Table 5).

| Parameter | Scenario 1 | Scenario 2 | Paper (Table 5) |
|---|---|---|---|
| Actor-critic hidden dim | 128 | 256 | 64 |
| N\_ENVS (parallel envs) | 8 | 16 | — |
| ROLLOUTS\_PER\_UPDATE | 8 | 16 | — |
| N\_EPISODES (update cycles) | 63 | 63 (+ extended) | ~97 |
| LR schedule | 3e-4 (flat) | 3e-4 → 1e-5 (linear) | 3e-4 → 0 |
| N\_EPOCHS per rollout | 10 | 20 | 10 |
| Minibatch size | 32 | 32 | 64 |
| GAMMA (β) | 0.99 | 0.99 | 0.99 |
| GAE\_LAMBDA | 0.95 | 0.95 | 0.95 |
| CLIP\_EPS | 0.2 | 0.2 | 0.2 |
| VF\_COEF | 0.5 | 0.5 | 0.5 |
| ENT\_COEF | 0.01 → 0.001 (linear) | same | not reported |
| MAX\_GRAD | 0.5 | 0.5 | 0.5 |
| REWARD\_SCALE | 20.0 | 20.0 | raw power |
| TERMINAL\_SCALE | 5.0 | 5.0 | — |
| CONVERGENCE\_COEF | 0.05 | 0.05 | — |
| Warm-start from Sc1 checkpoint | — | Yes (LOAD\_CHECKPOINT=True) | — |
| Inference action selection | greedy (actor\_mean, no sampling) | greedy (actor\_mean, no sampling) | — |
| Approximate total env steps | ~151 k | ~480 k | 200 k |

---

## Notable Cross-Developer Differences

- **Paradigm**: Zeph is the only RL developer. Kevin and Aleksei both fit surrogate models
  offline and then search for optimal yaw angles in a single optimization pass.

- **GP kernel**: Kevin uses sklearn's RBF kernel with restarts. Aleksei uses BoTorch's
  Matern(ν=0.5) with ARD length-scales — better suited to the anisotropic geometry of
  turbine wake interactions.

- **PFN scope**: Kevin uses TabPFN as a plug-in black-box regressor over a small context
  (10–32 episodes). Aleksei trains full in-context Transformer PFNs (PFNs4BO, KG-PFN) on
  100 k pre-simulated episodes, and adds an Axial PFN for multi-layout zero-shot transfer
  (Sc3).

- **Yaw search strategy at eval time**: Kevin samples 1024–2048 random candidate yaw
  vectors and scores each through the surrogate. Aleksei runs 100–200 steps of gradient
  descent via Adam through the differentiable surrogate. Zeph's PPO policy outputs yaw
  deltas directly as continuous actions — no explicit yaw search needed.

- **Data budget**: Kevin collects 10–32 live context episodes per evaluation run (~1.5–4.8 k
  steps). Aleksei pre-trains on 100 k offline simulations (~15 M simulated steps). Zeph
  accumulates 151 k–480 k online RL environment steps during training.

- **Scenario 3**: Only Aleksei implements a third scenario — multi-layout generalization via
  Axial PFN with leave-one-out cross-validation across layouts.

- **Reward / objective**: Zeph uses a shaped differential reward composed of three terms
  (scaled power gain, terminal bonus, convergence penalty). Kevin and Aleksei both optimize
  raw predicted farm power through their respective surrogates with no reward shaping.
