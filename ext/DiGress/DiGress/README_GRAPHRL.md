# Graph-RL Notes for DiGress on HOG Planar

This README is specific to the `graph-rl` and complements the upstream [README.md](./README.md).

It documents the workflow used in this repository for:

1. training the unconditional DiGress model on the HOG planar dataset,
2. training the four property regressors used for the Graph-RL use cases,
3. locating the checkpoints that are already present locally.

The commands and paths below were cross-checked against the local code, Hydra outputs, and local run artifacts in this checkout.

<!-- ## Training Logs and important files

In order to visualize training logs and load checkpoints files need to be downloaded from Google Drive at this link: 

Files could be moved directly to the outer DiGress directory. -->

## Working directory

Run the commands below from:

```bash
cd ext/DiGress/DiGress
```

The historical runs in this repository used the project virtualenv at:

```bash
../../.venv/bin/python3
```

## HOG dataset location

The HOG data lives under:

```bash
../hog_planar
```

In this checkout, the file that is present is:

```bash
../hog_planar/hog_planar_graph.pt
```
If `planar_graph.pt` already exists, do not recreate it.

## Train the unconditional HOG model

The HOG unconditional experiment is defined in:

```bash
configs/experiment/hog_planar_unconditional.yaml
```

A training run can be started with this command:

```bash
python3 src/main.py +experiment=hog_planar_unconditional
```

That experiment uses:

- dataset: `hog_planar`
- epochs: `1000`
- batch size: `8`
- learning rate: `2e-4`
- diffusion steps: `500`
- model name: `hog_planar_unconditional`

So the standard training command is:

```bash
cd ext/DiGress/DiGress
../../.venv/bin/python3 src/main.py +experiment=hog_planar_unconditional
```

Hydra writes outputs one level above this folder, under:

```bash
../outputs/<date>/<time>-hog_planar_unconditional
```

## Unconditional checkpoint used for the regressors

For the regressor and conditional sampling workflow in this folder, use the local checkpoint:

```bash
checkpoint/last.ckpt
```

This replaces the earlier reference to the checkpoint stored under `../outputs/...`.

## Train the property regressors

The regressor training entry point is:

```bash
scripts/train_regressor.py
```

This script builds metric targets directly from the HOG planar graphs using `src/datasets/hog_with_metrics.py`.

The available metric profiles are:

- `use_case_0`: `num_nodes`, `triangle_count`
- `use_case_1`: `num_nodes`, `spectral_gap`, `gini_coefficient`, `clustering_coefficient`
- `use_case_2`: `num_nodes`, `gini_coefficient`
- `use_case_3`: `num_nodes`, `spectral_gap`
- `all_4_use_cases`: all five metrics above in one regressor

For the four separate Graph-RL use-case regressors, run:

```bash
python scripts/train_regressor.py --metric_profile use_case_0 --epochs 200 --batch_size 32 --checkpoint_dir .
python scripts/train_regressor.py --metric_profile use_case_1 --epochs 200 --batch_size 32 --checkpoint_dir .
python scripts/train_regressor.py --metric_profile use_case_2 --epochs 200 --batch_size 32 --checkpoint_dir .
python scripts/train_regressor.py --metric_profile use_case_3 --epochs 200 --batch_size 32 --checkpoint_dir .
```

Using `--checkpoint_dir .` is important if you want the checkpoints written exactly under:

```bash
property_regressor/use_case_<k>/
```

## Regressor checkpoints already present locally

This repository already contains four use-case regressor checkpoint directories:

```bash
property_regressor/use_case_0
property_regressor/use_case_1
property_regressor/use_case_2
property_regressor/use_case_3
```

Each contains:

- `best_regressor.pt`
- `final_regressor.pt`

The checkpoints currently available are:

```bash
property_regressor/use_case_0/best_regressor.pt
property_regressor/use_case_1/best_regressor.pt
property_regressor/use_case_2/best_regressor.pt
property_regressor/use_case_3/best_regressor.pt
```

Use `best_regressor.pt` for sampling unless you specifically want the last epoch.

## Conditional sampling after training

The HOG conditional sampling script is:

```bash
scripts/sample_conditional_hog.py
```

Example with the local unconditional checkpoint and the use-case 1 regressor:

```bash
python scripts/sample_conditional_hog.py \
    --model_checkpoint checkpoint/last.ckpt \
    --regressor_checkpoint property_regressor/use_case_1/best_regressor.pt \
    --use_case use_case_1 \
    --num_samples 10 \
    --guidance_scale 2.0
```

Swap the regressor path and `--use_case` value for `use_case_0`, `use_case_2`, or `use_case_3` as needed.
