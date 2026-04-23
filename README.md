# Weather Vein

CSCI 5980 final project workspace for Prior Fitted Networks for contextual Bayesian
optimization on the wake steering problem. The project studies how surrogate models
and reinforcement learning methods can improve wind farm power production by steering
upstream turbine wakes away from downstream turbines.

Team:

- Aleksei Rozanov
- Kevin Babashov
- Zeph Johnson

## Project Context

The current project uses WFCRL to evaluate wind farm control policies. In this setup,
the controller chooses turbine yaw changes, and the simulator returns farm-level power
and turbine-level measurements.

Key context variables:

- Wind speed
- Wind direction
- Turbine coordinates
- Turbine yaw
- Turbine-level wind and load measurements

Current simulator focus:

- FLORIS is the current working simulator.
- FAST.Farm is the intended next simulator target if time permits.

Current scenario framing:

- Scenario 1: fixed wind speed of 8 m/s and fixed wind direction of 270 degrees.
- Scenario 2: randomized train-time initial conditions, with wind speed sampled from a
  Weibull distribution and wind direction sampled from a normal distribution; test-time
  evaluation remains fixed at 8 m/s and 270 degrees.
- Scenario 3: future expansion target if time permits.

Methods under comparison:

- Gaussian Process contextual Bayesian optimization
- TabPFN
- GraphPFN
- PPO / RL baselines
- Do-nothing and random baselines

TabPFN note: the active notebook uses the hosted Prior Labs client
`tabpfn_client.TabPFNRegressor` for reward/power regression. Store your token in
Colab Secrets as `TABPFN_TOKEN`, enable notebook access, and run the notebook's
TabPFN authentication cell before any `.fit()` call. If the token is missing or
invalid, the notebook uses a clearly labeled `TabPFN-Fallback` sklearn surrogate
so the rest of the FLORIS Scenario 2 pipeline can still run.

## Repo Map

- `CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_V2.ipynb`: active GraphPFN and TabPFN
  experiment notebook.
- `CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_Scenario1.ipynb`: Scenario 1 GraphPFN
  and TabPFN workflow.
- `CSCI_5980_notebooks/evaluate_scenarios.py`: Python runner that evaluates
  Scenario 1 and Scenario 2 together on FLORIS and writes comparable CSV outputs.
- `CSCI_5980_notebooks/GraphicalFailure_wfcrl.ipynb`: investigation of why raw graph
  inputs fail with TabPFN.
- `CSCI_5980_notebooks/WFCRL_GraphPFN_Synthetic.ipynb`: synthetic GraphPFN
  experiments.
- `wfcrl-env/`: WFCRL environment code, including FLORIS and FAST.Farm interfaces.
- `CSCI_5980_notebooks/__simul__/floris/`: generated FLORIS simulation cases.

## Setup

The WFCRL environment package is kept in `wfcrl-env/`.

```bash
pip install pandas scipy scikit-learn torch torch-geometric tabpfn-client python-dotenv
cd wfcrl-env
pip install -e .
```

To work with FAST.Farm, WFCRL expects the simulator dependencies to be available.
From inside `wfcrl-env/`, the upstream setup path is:

```bash
wfcrl-simulator fastfarm
python examples/example_fastfarm.py
```

## Running Scenario 1 and 2 Together

Use the Python runner when collecting final project results. It evaluates both
Scenario 1 and Scenario 2 for each selected FLORIS layout and seed, then writes:

- `CSCI_5980_notebooks/results/scenario_1_2_floris_results.csv`
- `CSCI_5980_notebooks/results/scenario_1_2_floris_summary.csv`
- `CSCI_5980_notebooks/results/method_counts.csv`

Quick smoke test:

```bash
python CSCI_5980_notebooks/evaluate_scenarios.py \
  --layouts Turb3_Row1_Floris \
  --seeds 0 \
  --n-initial 2 \
  --n-candidates 4 \
  --max-steps 5 \
  --graph-train-steps 1 \
  --cpu \
  --output-dir results_smoke
```

Full Kevin-branch FLORIS run:

```bash
python CSCI_5980_notebooks/evaluate_scenarios.py \
  --layouts slide \
  --seeds 0 1 2 \
  --n-initial 32 \
  --n-candidates 1024 \
  --max-steps 150 \
  --graph-train-steps 100
```

If `TABPFN_TOKEN` is available in the environment, the runner uses the hosted
Prior Labs TabPFN regressor. Otherwise it uses the same clearly labeled
`TabPFN-Fallback` sklearn surrogate so the FLORIS sweep can still finish.

## Actionables

1. Keep the current documentation and notebook work on the `Kevin` branch. Once the
   current rebase state is resolved, create or refresh a shared `dev` branch so graph
   results can be merged into one cohesive codebase for Aryan to examine.
2. Expand Kevin's GraphPFN and TabPFN work from Scenario 1 to Scenario 2.
   Implemented in `CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_V2.ipynb` for
   FLORIS by training on randomized Scenario 2 contexts and evaluating on the
   fixed 8 m/s, 270 degree test condition. The paired Scenario 1/2 evaluation
   runner is `CSCI_5980_notebooks/evaluate_scenarios.py`.
3. If time permits, switch the environment from FLORIS to FAST.Farm.
4. If time permits, switch or extend the experiments to Scenario 3.
5. Unify result collection across GP, TabPFN, GraphPFN, PPO, do-nothing, and random
   baselines so final report figures are comparable.
6. Prioritize final report and code work first, then the poster.
7. Complete check-ins independently for April 13 and April 27. As of April 20, 2026,
   the April 13 check-in is already past, and April 27 is the remaining check-in.

## Future Work

Aleksei:

- Expand experiments to Scenario 3 after completing Scenario 2.

Kevin:

- Expand GraphPFN and TabPFN to Scenario 2. The FLORIS notebook path is now in
  place; the next step is running the full layout/seed sweep and collecting CSV
  outputs.
- Work on switching the environment from FLORIS to FAST.Farm.

Zeph:

- Expand RL PPO to Scenarios 1 and 2.
- Work on switching the environment to FAST.Farm and/or trying Scenario 3.

Whole team:

- Finish cohesive code and final report work before the poster.
- Keep check-ins independent for April 13 and April 27.

## Experiment Notes

Scenario 1 GraphPFN and TabPFN results may underperform for several reasons:

- The graph structure may be weak, incomplete, or mismatched to the wake dependency
  structure.
- Context size may be too small for stable in-context behavior.
- Candidate ranking quality may limit yaw selection.
- GraphPFN training priors may not transfer cleanly from synthetic graph structure to
  WFCRL FLORIS layouts.

Scenario 2 is especially important because randomized initial conditions make wind
speed and wind direction true contextual variables. Contextual Bayesian optimization
and PFN-style posterior prediction should be evaluated there before drawing final
project conclusions.
