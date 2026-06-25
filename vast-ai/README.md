# Vast.ai Gemma4 RPP Training Package

This folder is a self-contained package for continuing the current Gemma4
Global RoutingPathPredictor training run on Vast.ai.

## Contents

- `experiments/gemma4_global_predictor/`: training code.
- `dataset/prompt10000/router_label_npz/npz/`: materialized NPZ samples.
- `checkpoints/current/checkpoint_best.pt`: latest available checkpoint snapshot
  from the active remote run.
- `checkpoints/resume/checkpoint_best.pt`: stable epoch-20 resume checkpoint.
- `Dockerfile`: CUDA-capable training image for hosts that can run Docker.
- `run_vast_train_docker.sh`: Docker entrypoint with GPU mount.
- `run_vast_train.sh`: direct in-container entrypoint for Vast.ai PyTorch images.
- `run_vast_grid_search.sh`: direct in-container grid-search entrypoint.
- `outputs/`: default output root for new training reports and checkpoints.

The base Gemma4 26B weights are intentionally not included. Training consumes
the pre-materialized NPZ files, which already contain token IDs, router logits,
top-k labels, masks, and metadata.

## Quick Start On Vast.ai

From this directory:

```bash
chmod +x run_vast_train.sh run_vast_train_docker.sh
./run_vast_train.sh
```

On the Vast.ai PyTorch template, `run_vast_train.sh` automatically prefers
`/venv/main/bin/python`, which is where the preinstalled CUDA-enabled PyTorch
environment usually lives. Override this with `PYTHON_BIN=/path/to/python` if
needed.

If the instance exposes Docker with NVIDIA runtime:

```bash
BUILD_IMAGE=1 ./run_vast_train_docker.sh
```

## Default Resume Configuration

The default command resumes from:

```text
checkpoints/current/checkpoint_best.pt
```

and writes a timestamped run under:

```text
outputs/rpp_train_vast_YYYYmmdd_HHMMSS/
```

Default hyperparameters mirror the active remote run:

- data: `dataset/prompt10000/router_label_npz/npz`
- epochs: `30`
- batch size: `8`
- max sequence length: `512`
- learning rate: `3e-4`
- embedding mode: `hash`
- KL weight: `0.0`
- split: stratified by `task_type,source`
- resume shuffle seed: derived from resumed epoch when `SHUFFLE_SEED=-1`

Override any setting with environment variables, for example:

```bash
BATCH_SIZE=16 EPOCHS=40 DEVICE=cuda ./run_vast_train.sh
```

## Grid Search On Vast.ai

By default this uses successive halving: all candidates run briefly, only the
top candidates continue, and the final report ranks by validation
`batch_level_accuracy@8`.

```bash
chmod +x run_vast_grid_search.sh
GRID_BATCH_SIZES=48 \
GRID_LRS=3e-4,1e-4 \
GRID_HEAD_HIDDEN_DIMS=128,256 \
GRID_POS_WEIGHTS=4.0,8.0,12.0,15.0,20.0,auto \
./run_vast_grid_search.sh
```

Default staged schedule:

- round 1: 24 candidates to 2 epochs
- round 2: top 6 candidates resume to 8 epochs
- round 3: top 2 candidates resume to 30 epochs

Controls:

- `GRID_HALVING_EPOCHS=1,6,20` changes the staged epoch schedule.
- `GRID_HALVING_KEEP=6,2` changes how many candidates are promoted.
- `GRID_SEARCH_MODE=grid GRID_EPOCHS=8` runs the original full grid mode.

The runner intentionally does not resume from `checkpoints/current/checkpoint_best.pt`
by default, because architecture search can change tensor shapes. To search only
compatible optimizer/training settings from an existing checkpoint, set:

```bash
GRID_RESUME=checkpoints/current/checkpoint_best.pt ./run_vast_grid_search.sh
```

Common controls:

- `GRID_MAX_RUNS=4` limits the number of combinations for a quick test.
- `GRID_OUT_ROOT=outputs/my_grid` sets the output root.
- `GRID_BATCH_SIZES=32,48`, `GRID_D_MODELS=32,64`,
  `GRID_ENCODER_LAYERS=2,3`, `GRID_DECODER_LAYERS=2,3`,
  `GRID_FFN_DIMS=2048,4096`, and `GRID_DROPOUTS=0.0,0.1` expand architecture
  search.
- `GRID_KL_WEIGHTS=0.0,0.05` and `GRID_KL_SCHEDULES=fixed,linear` search KL
  loss settings.

Each search writes:

- `grid_config.json`
- `grid_combinations.json`
- `grid_results.csv`
- `grid_results.json`
- `GRID_REPORT.md`
- `round*_e*/ROUND_REPORT.md` for successive-halving rounds
- one subdirectory per candidate run with its normal training outputs

## Expected Outputs

Each run writes:

- `config.json`
- `train_log.jsonl`
- `metrics.csv`
- `REPORT.md`
- `checkpoint_best.pt`
- `checkpoint_last.pt`
