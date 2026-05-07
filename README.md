# Weather Vein

Weather Vein is a research project for wind farm wake steering control. The
project studies whether prior fitted networks and related surrogate models can
optimize turbine yaw angles with fewer expensive simulator calls than standard
optimization or reinforcement learning approaches.

The experiments are built around the WFCRL wind farm control benchmark. In this
setting, a controller observes wind conditions, turbine layout information, and
turbine-level measurements, then selects yaw angles intended to improve total
farm power production.

## Motivation

Wind turbines create wakes: slower, more turbulent airflow behind upstream
turbines. These wakes can reduce the power produced by downstream turbines. Wake
steering attempts to reduce this loss by yawing upstream turbines so their wakes
are redirected away from downstream turbines.

This is a useful testbed for contextual Bayesian optimization because the reward
depends not only on the yaw vector, but also on the wind conditions and spatial
layout of the farm. Simulator calls are expensive, so methods that can learn from
small context sets or transfer across layouts are especially valuable.

## Methods

The repository compares several baselines and learning-based methods:

- Do-nothing baseline
- Random yaw baseline
- Gaussian process Bayesian optimization
- PFNs4BO
- TabPFN
- GraphPFN
- Axial PFN
- PPO reinforcement learning

The PFN-based methods are evaluated as surrogate models for predicting simulator
reward and selecting promising yaw configurations. Graph-based and axial variants
are included to better represent turbine-to-turbine wake structure and transfer
across layouts with different numbers of turbines.

## Repository Structure

- `CSCI_5980_notebooks/`: experiment notebooks and result-generation files
- `CSCI_5980_notebooks/results/`: CSVs, figures, and tables used in analysis
- `CSCI_5980_notebooks/evaluate_scenarios.py`: script for running comparable
  FLORIS scenario evaluations
- `wfcrl-env/`: local WFCRL environment code and simulator interfaces
- `kernel-wfcrl/`: Jupyter kernel configuration used for FAST.Farm experiments
- `.github/workflows/`: GitHub Actions for linting and testing
- `pyproject.toml`: project metadata and development tool configuration
- `LICENSE`: MIT license

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install the main Python dependencies:

```bash
pip install pandas numpy scipy scikit-learn matplotlib torch torch-geometric gpytorch tabpfn-client python-dotenv
```

Install the local WFCRL environment:

```bash
cd wfcrl-env
pip install -e .
cd ..
```

For TabPFN experiments, place your API token in a local `.env` file:

```bash
TABPFN_TOKEN=your_token_here
```

Do not commit `.env` files or API keys.

## Running Experiments

Most workflows are organized as Jupyter notebooks inside
`CSCI_5980_notebooks/`. The main FLORIS experiments use WFCRL layouts and compare
methods under fixed and randomized wind contexts.

A small smoke test can be run with:

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

Final runs use larger seed sets, more candidate yaw vectors, and 150 environment
steps.

## Results

Final result files are stored in `CSCI_5980_notebooks/results/`. These include
CSV summaries and bar plots comparing episode return and improvement over the
do-nothing baseline.

The final analysis suggests that PFN-based methods can be competitive across a
range of layouts and are useful as few-shot or zero-shot surrogate optimizers.
PPO performs strongly on several layouts where it was evaluated, but it requires
direct environment interaction during training. This makes PFNs attractive when
the goal is to reduce the number of online simulator calls.

## FAST.Farm and OpenFAST

The project also includes exploratory work with FAST.Farm through OpenFAST as a
higher-fidelity alternative to FLORIS. FAST.Farm is more physically detailed, but
the experiments were too computationally expensive for the available hardware and
project deadline. Some runs continued for multiple days without completing. The
FAST.Farm notebooks are included as a starting point for future work on larger
CPU or high-performance-computing resources.

## Development

Install development dependencies when working on scripts or package code:

```bash
pip install -e ".[dev]"
```

Run tests and quality checks with:

```bash
pytest
pre-commit run --all-files
```

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
