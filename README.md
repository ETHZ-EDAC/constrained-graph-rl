# Constrained, Grammar-based Graph Generation with Reinforcement Learning

This is the code base to NeurIPS 2026 paper titled 
`Constrained Goal-directed Planar Graph Generation with Grammar-based Reinforcement Learning`
by Nicolas Hochuli, Lorenzo Miele, Kristina Shea and Tino Stankovic.

## Installation

Install the dependencies with UV (make sure you have uv installed) by running the
command from the project root directory:
```bash
uv sync
```

## Usage
To train the proposed method, run the following command:
```bash
uv run scripts/train_rl.py
```
from the project root. The hyperparameters, constraints and target metrics for training can be modified in
- conf/policy/ppo.yaml
- conf/constraints/default.yaml
- conf/target_metrics/default.yaml

Be aware that the first rollout takes way longer than the subsequent ones due to jit compilation of the 
grammar in JAX.


### Baseline Evaluations
To evaluate baseline methods, run the following command:
```bash
uv run scripts/find_best_graphs_for_targets.py
```
Ensure to install the Boltzmann generator by following the instructions at `ext/boltzmann-planar-graphs` using 
`uv pip install -e .`.

To run deep generative baselines, i.e. DiGress, GDSS with PRODIGY (in `code` directory) and MuDiff, see the respective 
Readme files  in the `ext` directory.

## Citation

If you use this code or method please cite us using:

```bibtex
@inproceedings{
hochuli2026constrained,
title={Constrained Goal-directed Planar Graph Generation with Grammar-based Reinforcement Learning},
author={Hochuli, Nicolas and Miele, Lorenzo and Shea, Kristina and Stankovic, Tino},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=9Sh3SI4EF1}
}
```

## License