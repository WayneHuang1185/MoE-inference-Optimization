# Per-Memory Expert Capacity Estimate

Estimate how many Gemma4 MoE experts remain page-cache resident when decode KV
cache grows from the measured 5-token baseline to longer decode lengths.

The estimator is non-invasive: it reuses existing
`expert_cache_matrices.json` files from the current memory experiment plus
`gemma4_26b_tensor_ranges.csv`. This folder is intended to live under
`mem8G/expert_capacity/` or `mem10G/expert_capacity/`.

## Run

```bash
rsync -av experiments/gemma4_IO_behaviors/mem8G/expert_capacity/ \
  nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_IO_behaviors/mem8G/expert_capacity/

rsync -av experiments/gemma4_IO_behaviors/mem10G/expert_capacity/ \
  nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_IO_behaviors/mem10G/expert_capacity/
```

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && \
  experiments/gemma4_IO_behaviors/mem8G/expert_capacity/run_expert_capacity_estimate.sh'

ssh nthu-cs 'cd ~/workspace/HuangWayne/project && \
  experiments/gemma4_IO_behaviors/mem10G/expert_capacity/run_expert_capacity_estimate.sh'
```

## Outputs

- `statistics/capacity_estimate_<timestamp>/capacity_predictions_by_sample.csv`
- `statistics/capacity_estimate_<timestamp>/capacity_summary.csv`
- `statistics/capacity_estimate_<timestamp>/token_abs_diff_projection.csv`
- `statistics/capacity_estimate_<timestamp>/token_abs_diff_projection_summary.csv`
- `statistics/capacity_estimate_<timestamp>/run_config.json`
- `figures/capacity_estimate_<timestamp>/capacity_abs_diff_histogram.svg`
- `figures/capacity_estimate_<timestamp>/capacity_abs_diff_summary_bars.svg`
- `figures/capacity_estimate_<timestamp>/capacity_abs_diff_by_token.svg`

By default the runner parses the current memory environment's latest
`statistics/decode_expert_cache_fault_*/server.log` and derives KV bytes per
token from the logged KV cache MiB/cell allocation. Set `KV_BYTES_PER_TOKEN` to
override this calibration.
