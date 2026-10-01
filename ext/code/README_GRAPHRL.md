# GraphRL Workflow for GDSS

This note describes the GDSS workflow used in this repository: how to set up the environment, train the base GDSS model, and run constrained sampling with `PRODIGY`.

It is specific to the code under `ext/code/`, especially:

- `models/GDSS/` for the underlying diffusion model
- `prodigy.py` for constrained sampling

<!-- ## Training Logs and important files

In order to visualize training logs and load checkpoints files need to be downloaded from Google Drive at this link: 

Files could be moved directly to the GDSS directory. -->

## 1. Install dependencies

From `ext/code/`, install the shared package used by the controllable sampling code:

```bash
cd ext/code
python setup.py install
```

Then install the GDSS dependencies:

```bash
cd models/GDSS
pip install -r requirements.txt
```

If you want generic graph generation metrics available in the environment, compile ORCA once:

```bash
cd evaluation/orca
g++ -O2 -std=c++11 -o orca orca.cpp
```

## 2. Train the base GDSS model

All native GDSS training commands are run from `ext/code/models/GDSS`.

The training configuration used here is `config/hog_planar.yaml`. It points to the `hog_planar_lt100` dataset and saves checkpoints every 200 epochs.

In order to train the model run:

```bash
cd ext/code/models/GDSS
CUDA_VISIBLE_DEVICES=0 python main.py --type train --config hog_planar --seed 42
```

A useful Training checkpoint used here is:
- checkpoint: `Mar26-22:04:21_800`
Where the model trained for 800 epochs with good metrics using the standard GDSS config (`config/hog_planar.yaml`)

Useful files:

- `config/hog_planar.yaml`: training configuration
- `checkpoints/hog_planar_lt100/`: saved checkpoints
- `logs_train/hog_planar_lt100/hog_planar_train/`: training logs

## 3. Check the GDSS sampling config

Constrained runs in this repo use GDSS through `PRODIGY`, but they still rely on the usual GDSS sampling configuration and checkpoint naming.

The main sampling config is:

```text
ext/code/models/GDSS/config/sample_hog_planar.yaml
```

It currently uses:

- dataset: `hog_planar_lt100`
- checkpoint: `Mar26-22:04:21_800`

This is the intended single GDSS sampling config for the HOG planar workflow. The node-count target belongs in the constraint file, not in separate sampling-config filenames.

## 4. Run constrained sampling with PRODIGY

Run constrained sampling from `ext/code`.

Example:

```bash
cd ext/code
python prodigy.py \
  --model GDSS \
  --dataset hog_planar \
  --constraint configs/nnodes/use_cases/constraint_use_case_0_rectangularity_60_box.yaml \
  --method configs/nnodes/method.yaml \
  --device cuda:0
```

What each argument controls:

- `--model GDSS`: uses the GDSS backend in `models/GDSS/`
- `--dataset hog_planar`: uses the canonical GDSS sample config
- `--constraint ...`: defines the target constraint
- `--method ...`: defines the projection schedule and other PRODIGY settings
- `--device cuda:0`: selects the device

The use case, including the desired node-count range such as 40, 50, or 60 nodes, is controlled by the chosen constraint YAML.

Current HOG planar use cases in `configs/nnodes/use_cases/`:

- `constraint_use_case_0_rectangularity_60_box.yaml`: `num_nodes=60`, `isoperimetric_ratio=1.0`, `triangle_count=40`
- `constraint_use_case_1_power_grid_50_box.yaml`: `num_nodes=50`, `spectral_gap=1.0`, `gini_coefficient=0.1`, `isoperimetric_ratio=1.0`, `clustering_coefficient=0.2`
- `constraint_use_case_2_urban_80_box.yaml`: `num_nodes=80`, `angular_resolution=0.8`, `edge_length_deviation=0.7`, `gini_coefficient=0.1`, `isoperimetric_ratio=1.0`
- `constraint_use_case_3_aircraft_100_box.yaml`: `num_nodes=100`, `spectral_gap=0.3`, `rectangularity=1.0`

Relevant files:

- `prodigy.py`: entry point for constrained sampling
- `configs/nnodes/method.yaml`: method configuration
- `configs/nnodes/use_cases/`: ready-made constraint files

## 5. Best-graph reporting during sampling

For the workflow used here, the best graph was reported during sampling itself rather than through a separate `evaluate.py` pass.

The relevant artifacts are written under:

- `models/GDSS/logs_sample/`
- `models/GDSS/samples/pkl/`
- `models/GDSS/samples/best_graph_eval_case_*`

The `best_graph_eval_case_*` folders contain the saved best graph, best planar graph, and the corresponding JSON summaries.

## 6. Where outputs are written

During training, GDSS writes:

- checkpoints to `models/GDSS/checkpoints/`
- logs to `models/GDSS/logs_train/`

During sampling, GDSS and PRODIGY write:

- sample logs to `models/GDSS/logs_sample/`
- sampled graph pickles to `models/GDSS/samples/pkl/`
- selected best-graph visualizations and stats to `models/GDSS/samples/best_graph_eval_case_*`

## 7. Figures by use case

### Use case 0: rectangularity / 60-node target

This case uses the rectangularity-style setup with the `constraint_use_case_0_rectangularity_60_box.yaml` constraint. The saved outputs below show the best graph found during sampling and the best planar graph variant saved for this run.

Best graph:

![Use case 0 best graph](models/GDSS/samples/best_graph_eval_case_0_v3/hog_planar_lt100_Mar26-22:04:21_800_best_graph.png)

Best planar graph:

![Use case 0 best planar graph](models/GDSS/samples/best_graph_eval_case_0_v3/hog_planar_lt100_Mar26-22:04:21_800_best_planar_graph.png)

### Use case 1: power-grid target

This case uses the power-grid-style target from `constraint_use_case_1_power_grid_50_box.yaml`. As above, both the best unconstrained match and the best planar graph saved during sampling are shown.

Best graph:

![Use case 1 best graph](models/GDSS/samples/best_graph_eval_case_1_v3/hog_planar_lt100_Mar26-22:04:21_800_best_graph.png)

Best planar graph:

![Use case 1 best planar graph](models/GDSS/samples/best_graph_eval_case_1_v3/hog_planar_lt100_Mar26-22:04:21_800_best_planar_graph.png)

### Use case 2: urban target

This case uses the urban-style target from `constraint_use_case_2_urban_80_box.yaml`. The saved figures correspond to the best graph selected during the sampling run and its best planar alternative.

Best graph:

![Use case 2 best graph](models/GDSS/samples/best_graph_eval_case_2_v3/hog_planar_lt100_Mar26-22:04:21_800_best_graph.png)

Best planar graph:

![Use case 2 best planar graph](models/GDSS/samples/best_graph_eval_case_2_v3/hog_planar_lt100_Mar26-22:04:21_800_best_planar_graph.png)

### Use case 3: aircraft target

This case uses the aircraft-style target from `constraint_use_case_3_aircraft_100_box.yaml`. The figures below show the best graph reported during sampling and the best planar graph saved for the same use case.

Best graph:

![Use case 3 best graph](models/GDSS/samples/best_graph_eval_case_3_v3/hog_planar_lt100_Mar26-22:04:21_800_best_graph.png)

Best planar graph:

![Use case 3 best planar graph](models/GDSS/samples/best_graph_eval_case_3_v3/hog_planar_lt100_Mar26-22:04:21_800_best_planar_graph.png)
