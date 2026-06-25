# Gemma4 Global Predictor

Standalone implementation of a paper-style Global RPP for Gemma4 26B.

The code lives outside `experiments/gemma4_bottleneck` to keep the training
implementation separate from the bottleneck/inference utilities.

## Directory Format

- `statistics/`: remote-only CSV, JSON, and Markdown experiment reports.
- `figures/`: local/remote experiment data images only.
- `utils/`: reserved for reusable sidecar tools.
- Python modules in this directory implement the RPP dataset, model, training,
  and evaluation entry points.

## Data

Default remote data root:

```text
dataset/prompt1000/router_label_npz/npz
```

Each sample must contain:

- `input_ids [S]`
- `router_logits [S,30,128]`
- `router_topk [S,30,8]`
- `loss_mask [S]`
- `layer_mask [30]`
- `meta_json`

Only `loss_mask=1` tokens contribute to BCE/KL loss.

## Model

Default model is a lightweight T5-style encoder-decoder:

- Gemma4 tokenizer vocab aligned to `VOCAB_SIZE=262144`
- hashed token embedding by default: `EMBEDDING_MODE=hash`,
  `HASH_VOCAB_SIZE=32768`
- learned encoder positions
- Transformer encoder over `input_ids`
- Transformer decoder with learned position queries
- 30 independent expert heads producing `[B,S,30,128]`
- optional per-layer MLP heads via `HEAD_HIDDEN_DIM>0`
- defaults: `d_model=32`, `ffn_dim=2048`, 2 encoder layers, 2 decoder layers

Use `EMBEDDING_MODE=full` only for an ablation that gives every Gemma4 token id
its own embedding row. With `d_model=32`, full embedding is about 8.39M
parameters, while the default hash table is about 1.05M parameters.

Long sequences are tail-cropped by default with `MAX_SEQ_LEN=512`, preserving
generated-output labels at the end of each sample.

## Split

Training defaults to a deterministic stratified split over `task_type,source`:

- train: `TRAIN_FRAC=0.8`
- holdout: `0.2`
- validation: one fold from the holdout, controlled by `VAL_FOLDS=5` and
  `VAL_FOLD_INDEX=0`
- test: the remaining holdout folds

With the prompt10000 dataset this gives 8000 train, 400 validation, and 1600
test samples by default. Use `VAL_FOLD_INDEX=0..4` to rotate validation folds.
Set `SPLIT_STRATEGY=legacy` only to reproduce the old sorted 90/5/5 split.

## Loss

```text
loss = BCEWithLogits(pred, topk_membership) + 0.1 * KL(true || pred)
```

Defaults:

- `POS_WEIGHT=auto`, resolved to `(128 - 8) / 8 = 15.0`
- `KL_SCHEDULE=fixed`
- `KL_WEIGHT=0.1`
- optional scheduled KL: `KL_SCHEDULE=quadratic`, `KL_START=0.05`,
  `KL_END=0.5`, `KL_WARMUP_RATIO=0.35`
- `TEMPERATURE=1.0`

The routing target is multi-label top-k membership, not single-label
cross-entropy. For Gemma4 each MoE layer has 128 experts and 8 positives, so
`pos_weight` compensates for the 120 negative experts per layer.

When KL scheduling is enabled, logs include both raw `kl` and
`weighted_kl = kl_weight * kl` so BCE/top-k learning and distribution
distillation can be inspected separately.

## Remote Run

Sync only the code, not `dataset/`:

```bash
rsync -av experiments/gemma4_global_predictor/ \
  nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_global_predictor/
```

Build image and run a small smoke train:

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && \
  BUILD_IMAGE=1 MAX_FILES=8 EPOCHS=1 BATCH_SIZE=2 LOG_EVERY=1 \
  ./experiments/gemma4_global_predictor/run_train_rpp_docker.sh'
```

Full pilot:

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && \
  MAX_FILES=0 EPOCHS=5 BATCH_SIZE=4 MAX_SEQ_LEN=512 \
  ./experiments/gemma4_global_predictor/run_train_rpp_docker.sh'
```

Run another validation fold:

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && \
  DATA_ROOT=dataset/prompt10000/router_label_npz/npz \
  TRAIN_FRAC=0.8 VAL_FOLDS=5 VAL_FOLD_INDEX=1 \
  MAX_FILES=0 EPOCHS=5 BATCH_SIZE=8 MAX_SEQ_LEN=512 KL_WEIGHT=0.0 \
  ./experiments/gemma4_global_predictor/run_train_rpp_docker.sh'
```

Monitor latest run:

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && \
  python3 experiments/gemma4_global_predictor/monitor_rpp_train.py'
```

## Decode Layer Target Recall

Compare top2/top4/top6/top8/top16 target recall across MoE layers for 1000 prompts. The
target recall is `hits / min(k, 8)` for each token/layer, then aggregated by layer. Each
prompt contributes the first 5 consecutive completion/decode tokens from
`loss_mask=1`.

```bash
rsync -av experiments/gemma4_global_predictor/ \
  nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_global_predictor/
```

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && \
  MAX_SAMPLES=1000 DECODE_TOKENS=5 TOPKS=2,4,6,8,16 \
  ./experiments/gemma4_global_predictor/run_decode_layer_precision_docker.sh'
```

Outputs:

- `statistics/decode_layer_precision_<timestamp>/layer_topk_precision.csv`
- `statistics/decode_layer_precision_<timestamp>/decode_position_layer_topk_precision.csv`
- `statistics/decode_layer_precision_<timestamp>/summary.json`
- `figures/decode_layer_precision_<timestamp>/decode_layer_topk_precision.svg`
- `figures/decode_layer_precision_<timestamp>/decode_layer_topk_precision_heatmap.svg`
