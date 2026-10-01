# Graph-RL Notes for MuDiff on HOG Planar

This README is specific to the `graph-rl` MuDiff fork and complements the existing [readme.md](./readme.md).

It documents the planar HOG workflow used in this repository for the four Graph-RL use cases:

1. training MuDiff on planar HOG graphs with target-conditioned channels,
2. locating the concrete checkpoints already present locally,
3. reusing those checkpoints for target-conditioned sampling.

## Working directory

Run the commands below from:

```bash
cd ext/MuDiff/mudiff
```

The historical runs in this repository used the project virtualenv at:

```bash
../../.venv/bin/python
```


<!-- ## Training Logs and important files

In order to visualize training logs and load checkpoints files need to be downloaded from Google Drive at this link: 

Files could be moved directly to the mudiff directory. -->

## Dataset

The planar HOG data used by MuDiff is the file:

```bash
planar_graph.pt
```

## Where MuDiff saves checkpoints

MuDiff writes all run artifacts under:

```bash
outputs/<exp_name>/
```

The relevant checkpoint files are:

- `generative_model.npy`: best validation checkpoint
- `generative_model_ema.npy`: EMA version of the best validation checkpoint
- `generative_model_last.npy`: latest checkpoint saved during training
- `generative_model_ema_last.npy`: latest EMA checkpoint
- `optim.npy`, `optim_last.npy`
- `args.pickle`, `args_last.pickle`

This matches the logic in `main_qm9.py`: the training loop updates `generative_model.npy` and `args.pickle` when validation improves, and also writes `*_last.npy` snapshots.

### NOTE

Some changes were done to MuDiff to adapt main_qm9.py to a 2d case for palanr graphs. These chagnes include the setting to 0 on the third dimension (script was built with 3d molecules in mind)

## General training pattern

The command used here is:

```bash
../../.venv/bin/python main_qm9.py \
    --exp_name rectangularity_v2 \
    --wandb_run_name rectangularity_v2 \
    --dataset planar \
    --datadir planar_graph.pt \
    --n_dims 2 \
    --dp False \
    --save_model True \
    --conditioning_use_case constraint_use_case_0_rectangularity_60_box.yaml \
    --conditioning_use_case_dir configs/conditioning/use_cases \
    --n_epochs 400 \
    --batch_size 2 \
    --diffusion_steps 500 \
    --lr 5e-5 \
    --test_epochs 30 \
    --wandb True \
    --wandb_mode online \
    --wandb_project mudiff-geometric \
    --wandb_log_conditioning_metrics True \
    --conditioning_metric_eval_samples 4 \
    --best_graph_relative_error_eps 1e-6 \
    --wandb_log_target_conditioned_samples True \
    --wandb_target_num_sample_graphs 4 \
    --wandb_target_sample_every 10
```

Later saved runs in this checkout use the same overall pattern, but with final experiment names and explicit `--track_best_graphs True`.

## The four Graph-RL use cases

The four use-case YAML files are:

```bash
configs/conditioning/use_cases/constraint_use_case_0_rectangularity_60_box.yaml
configs/conditioning/use_cases/constraint_use_case_1_power_grid_50_box.yaml
configs/conditioning/use_cases/constraint_use_case_2_urban_50_box.yaml
configs/conditioning/use_cases/constraint_use_case_3_aircraft_40_box.yaml
```

Their target metrics are:

- `use_case_0` rectangularity: `num_nodes=60`, `isoperimetric_ratio=1.0`, `triangle_count=40`
- `use_case_1` power grid: `num_nodes=50`, `spectral_gap=1.0`, `gini_coefficient=0.1`, `isoperimetric_ratio=1.0`, `clustering_coefficient=0.2`
- `use_case_2` urban: `num_nodes=50`, `angular_resolution=0.8`, `edge_length_deviation=0.7`, `gini_coefficient=0.1`, `isoperimetric_ratio=1.0`
- `use_case_3` aircraft: `num_nodes=40`, `spectral_gap=0.3`, `rectangularity=1.0`

### Use case 1: power grid

Saved run:

```bash
outputs/power_grid_v10
```

Training command:

```bash
../../../.venv/bin/python main_qm9.py \
    --exp_name power_grid_v10 \
    --wandb_run_name power_grid_v10 \
    --dataset planar \
    --datadir planar_graph.pt \
    --n_dims 2 \
    --dp False \
    --save_model True \
    --conditioning_use_case constraint_use_case_1_power_grid_50_box.yaml \
    --conditioning_use_case_dir configs/conditioning/use_cases \
    --n_epochs 600 \
    --batch_size 2 \
    --diffusion_steps 500 \
    --lr 5e-5 \
    --test_epochs 30 \
    --wandb True \
    --wandb_mode online \
    --wandb_project mudiff-geometric \
    --wandb_log_conditioning_metrics True \
    --conditioning_metric_eval_samples 16 \
    --track_best_graphs True \
    --best_graph_eval_samples 16 \
    --best_graph_save_every 10 \
    --best_graph_relative_error_eps 1e-6 \
    --wandb_log_target_conditioned_samples True \
    --wandb_target_num_sample_graphs 16 \
    --wandb_target_sample_every 10 \
    --run_stability_eval False
```

### Use case 2: urban

Saved run:

```bash
outputs/urban_v12
```

Training command:

```bash
../../../.venv/bin/python main_qm9.py \
    --exp_name urban_v12 \
    --wandb_run_name urban_v12 \
    --dataset planar \
    --datadir planar_graph.pt \
    --n_dims 2 \
    --dp False \
    --save_model True \
    --conditioning_use_case constraint_use_case_2_urban_50_box.yaml \
    --conditioning_use_case_dir configs/conditioning/use_cases \
    --n_epochs 600 \
    --batch_size 2 \
    --diffusion_steps 500 \
    --lr 5e-5 \
    --test_epochs 30 \
    --wandb True \
    --wandb_mode online \
    --wandb_project mudiff-geometric \
    --wandb_log_conditioning_metrics True \
    --conditioning_metric_eval_samples 8 \
    --track_best_graphs True \
    --best_graph_eval_samples 8 \
    --best_graph_save_every 10 \
    --best_graph_relative_error_eps 1e-6 \
    --wandb_log_target_conditioned_samples True \
    --wandb_target_num_sample_graphs 8 \
    --wandb_target_sample_every 10 \
    --run_stability_eval False
```

### Use case 3: aircraft

Saved run:

```bash
outputs/aircraft_v3
```

Training command:

```bash
../../../.venv/bin/python main_qm9.py \
    --exp_name aircraft_v3 \
    --wandb_run_name aircraft_v3 \
    --dataset planar \
    --datadir planar_graph.pt \
    --n_dims 2 \
    --dp False \
    --save_model True \
    --conditioning_use_case constraint_use_case_3_aircraft_40_box.yaml \
    --conditioning_use_case_dir configs/conditioning/use_cases \
    --n_epochs 400 \
    --batch_size 2 \
    --diffusion_steps 500 \
    --lr 5e-5 \
    --test_epochs 30 \
    --wandb True \
    --wandb_mode online \
    --wandb_project mudiff-geometric \
    --wandb_log_conditioning_metrics True \
    --conditioning_metric_eval_samples 4 \
    --track_best_graphs True \
    --best_graph_eval_samples 4 \
    --best_graph_save_every 10 \
    --best_graph_relative_error_eps 1e-6 \
    --wandb_log_target_conditioned_samples True \
    --wandb_target_num_sample_graphs 4 \
    --wandb_target_sample_every 10 \
    --run_stability_eval False
```
## Target-conditioned sampling from a saved checkpoint

For post-training target-conditioned sampling, use:

```bash
sample_target_conditioned_wandb.py
```

Example for the saved power-grid checkpoint:

```bash
../../../.venv/bin/python sample_target_conditioned_wandb.py \
    --checkpoint_dir outputs/power_grid_v10 \
    --checkpoint_name generative_model_ema.npy \
    --conditioning_use_case constraint_use_case_1_power_grid_50_box.yaml \
    --conditioning_use_case_dir configs/conditioning/use_cases \
    --num_samples 8 \
    --fix_noise True \
    --wandb True \
    --wandb_project mudiff-geometric \
    --wandb_mode offline
```

Swap the checkpoint directory and use-case YAML for the other three use cases.

## Notes

- MuDiff planar HOG runs in this repo are 2D runs: `--n_dims 2`.
- The batch size used in the saved runs is low '2' runs on a RTX 5090, an 8 could be run on a RTX 6000 Blackwell.
- The diffusion step count in the saved runs is consistently `500`.
- The learning rate in the saved runs is consistently `5e-5`.
- Experiments were trained for 600 Epochs, even though the best conditioned samples were achieved consistently before the 200 epochs
