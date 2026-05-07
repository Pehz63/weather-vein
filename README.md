# Weather Vein

Weather Vein is a CSCI 8980 final project studying wake steering for wind farm
control. The project compares classical Bayesian optimization, Prior Fitted
Network based surrogates, and reinforcement learning baselines on the WFCRL wind
farm control benchmark.

The central question is whether PFN-style models can provide useful few-shot or
zero-shot optimization behavior across wind farm layouts, reducing the number of
expensive simulator calls needed to find strong turbine yaw settings.

## Project Overview

Wind turbines can reduce downstream power production by creating wakes: slower,
more turbulent wind behind upstream turbines. Wake steering adjusts turbine yaw
angles so upstream wakes are redirected away from downstream turbines.

This repository evaluates wake steering as a contextual Bayesian optimization
problem. Given a wind farm layout, wind conditions, turbine measurements, and a
candidate yaw vector, each method attempts to maximize farm-level reward.

Methods compared in this project include:

- Do-nothing and random baselines
- Gaussian process Bayesian optimization
- PFNs4BO
- TabPFN
- GraphPFN
- Axial PFN
- PPO reinforcement learning

The main simulator used for final results is FLORIS through the WFCRL
environment. FAST.Farm/OpenFAST experiments are included as exploratory work, but
were too computationally expensive for the full final sweep.

## Repository Structure

- `CSCI_5980_notebooks/`: main experiment notebooks and generated results
- `CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_V2.ipynb`: GraphPFN and TabPFN
  Scenario II workflow
- `CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_Scenario1.ipynb`: Scenario I
  GraphPFN and TabPFN workflow
- `CSCI_5980_notebooks/WFCRL_PFNs_FastFarm.ipynb`: exploratory FAST.Farm
  notebook
- `CSCI_5980_notebooks/WFCRL_PFNs_FastFarm_Windows.ipynb`: Windows-oriented
  FAST.Farm setup notebook
- `CSCI_5980_notebooks/evaluate_scenarios.py`: script for running comparable
  FLORIS evaluations
- `CSCI_5980_notebooks/results/`: result CSVs and final figures
- `wfcrl-env/`: local WFCRL environment code and simulator interfaces
- `kernel-wfcrl/`: Jupyter kernel configuration used for FAST.Farm experiments

## Setup

Create a Python environment and install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scipy scikit-learn matplotlib torch torch-geometric gpytorch tabpfn-client python-dotenv
```

Install the local WFCRL environment:

```bash
cd wfcrl-env
pip install -e .
cd ..
```

For TabPFN runs, store your Prior Labs API token in a local `.env` file:

```bash
TABPFN_TOKEN=your_token_here
```

Do not commit `.env` or API keys to GitHub.

## Running Experiments

Most experiments were run from Jupyter notebooks in `CSCI_5980_notebooks/`.

For a quick FLORIS smoke test using the Python runner:

```bash
python CSCI_5980_notebooks/evaluate_scenarios.py \
  --layouts Turb3_Row1_Floris \
  --seeds 0 \
  --n-initial 2 \
  --n-candidates 4 \
  --max-steps 5 \
  --graph-train-steps 1 \
  --cpu \
  --output-dir CSCI_5980_notebooks/results_smoke
```

For larger final runs, use more layouts, seeds, candidate yaw vectors, and
environment steps. The final FLORIS experiments use 150 environment steps.

## Results

Final result artifacts are stored in `CSCI_5980_notebooks/results/`, including:

- Scenario II episode-return tables
- Scenario II improvement-over-do-nothing figures
- CSV files used to generate the paper tables and plots

The final analysis found that PFN-based methods are competitive across several
layouts, with PFN variants winning among non-baseline methods on many layouts.
PPO is stronger on several layouts where it was evaluated, but it requires direct
environment interaction during training, while PFN-based approaches are intended
to reduce online simulator usage.

## FAST.Farm Notes

FAST.Farm through OpenFAST was tested as a higher-fidelity alternative to FLORIS.
This direction is scientifically valuable because FAST.Farm models wind farm
dynamics in greater physical detail. In practice, the runtime was too large for
the deadline and available hardware, with some runs continuing for multiple days
without completing. The FAST.Farm notebooks are therefore included as exploratory
work and a starting point for future high-performance-computing runs.

## Team

- Aleksei Rozanov
- Kevin Babashov
- Zephaniah Johnson

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Citation

If citing this repository, use:

```bibtex
@misc{weathervein2026,
  author = {Rozanov, Aleksei and Babashov, Kevin and Johnson, Zephaniah},
  title = {Weather Vein: Prior Fitted Networks for Wake Steering Control},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Pehz63/weather-vein}}
}
```
