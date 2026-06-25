# Global MoE Router Predictor — Data Collection

ExpertFlow-style global predictor for Gemma4 26B (L=30 MoE layers, E=128
experts/layer, K=8 top-k, hidden=2816).

**Goal**: collect `(input_ids[S], router_logits[S, L, E], router_topk[S, L, K])`
tuples so we can train a `input_ids → all-layer expert selection` model.

## Pilot (100 prompts)

Everything is non-invasive — we only consume the existing `activation_dump/`
files emitted by `run_container_ram_case.sh` (`GGML_ACTIVATION_DUMP_DIR`).

```bash
# on nthu-cs, inside the project dir
experiments/gemma4_bottleneck/run_global_predictor_dump_pilot.sh
```

Defaults (override with env vars):

| Var | Default | Notes |
|---|---|---|
| `MAX_PROMPTS` | 100 | reads from `router_prediction_prompts/` |
| `N_PREDICT` | 1 | prefill + 1 decode token; we keep only the prefill pass |
| `CTX_SIZE` | 8192 | matches existing bench scripts |
| `MODEL` | `models/gemma4-26B.gguf` | |
| `IMAGE` | `localhost/gemma4-ram-bench:24.04` | reuse existing podman image |

Outputs land under
`experiments/gemma4_bottleneck/results/global_predictor_pilot_<ts>/`:

```
global_predictor_pilot_<ts>/
├── router_prediction_batch/        # raw activation dumps (~5–10 GB)
│   ├── manifest.csv
│   └── <prompt_id>/activation_dump/*.bin
├── dataset/
│   ├── npz/<prompt_id>.npz         # per-prompt consolidated arrays
│   ├── dataset_manifest.csv
│   ├── dataset_summary.json
│   └── REPORT.md
└── FOOTPRINT.md
```

### Per-prompt `.npz` layout

| key | dtype | shape | meaning |
|---|---|---|---|
| `input_ids` | int32 | `[S]` | from `llama-tokenize -m <model> -f <prompt> --ids` |
| `router_logits` | float16 | `[S, L, E]` | `ffn_moe_logits-{i}` per layer |
| `router_topk` | int8 | `[S, L, K]` | argpartition top-K of `router_logits` |
| `layer_mask` | uint8 | `[L]` | 1 if `ffn_moe_logits-{i}` was captured |
| `meta_json` | uint8 | utf-8 bytes | per-prompt metadata |

```python
import numpy as np
d = np.load("00_moe_intro.npz")
ids    = d["input_ids"]        # int32 [S]
logits = d["router_logits"]    # fp16  [S, 30, 128]
topk   = d["router_topk"]      # int8  [S, 30, 8]
```

### What this pilot tells us

1. **per-prompt latency** (wall time / 100 ≈ avg sec/prompt) — for extrapolation.
2. **`logit_absmax_global`** — confirms fp16 has headroom (target < 100).
3. **`layers_captured_total / expected_layers_total`** — should be 100 % unless
   the dump path skipped a layer.
4. **per-prompt npz size** — extrapolation point for mid (1k) / full (5k+) runs.

### Loss recipe for the predictor (next step, after data)

```python
# multi-label BCE: 8 / 128 positives per (token, layer)
pos_weight = (E - K) / K    # ≈ 15.0
bce = F.binary_cross_entropy_with_logits(
    pred_logits, topk_mask, pos_weight=torch.tensor(pos_weight))

# distillation: KL(teacher || student) on softmax over experts
log_pred = F.log_softmax(pred_logits, dim=-1)
log_true = F.log_softmax(true_logits.float(), dim=-1)
kl = (log_true.exp() * (log_true - log_pred)).sum(-1).mean()

loss = bce + lam_kl * kl
# JS divergence is a clean ablation against `kl` (set lam_kl = 0 and use js).
```

## Local → Sync → Remote (CLAUDE.md §2)

```bash
# 1. local: make sure scripts are committed/staged
git status experiments/gemma4_bottleneck/

# 2. sync to nthu-cs (matches existing project layout)
rsync -av --delete \
  --exclude='results/' --exclude='__pycache__/' \
  experiments/gemma4_bottleneck/ \
  nthu-cs:~/project/experiments/gemma4_bottleneck/

# 3. remote: kick the pilot
ssh nthu-cs 'cd ~/project && experiments/gemma4_bottleneck/run_global_predictor_dump_pilot.sh'

# (later) pull only the consolidated dataset back, not the raw dumps
rsync -av nthu-cs:~/project/experiments/gemma4_bottleneck/results/global_predictor_pilot_*/dataset/ \
  experiments/gemma4_bottleneck/results/global_predictor_pilot_local/
```

## Scaling out (after pilot validates)

Mid (1 000 prompts): add 900 more prompts to `router_prediction_prompts/` (or
point `PROMPT_DIR` at a larger curated set) and re-run. Expected total npz
~2 GB at fp16.

Full (5 000 prompts): same, ~10 GB at fp16. The raw activation dump grows
linearly too — plan to clean up `router_prediction_batch/` per prompt as soon
as its `.npz` is written (a future flag in `prepare_global_predictor_dataset.py`).
