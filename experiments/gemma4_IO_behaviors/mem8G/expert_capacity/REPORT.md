# Expert Capacity Predicted vs Observed

## Config

- baseline_decode_tokens: `5`
- decode_targets: `5,50,100,200,300,400,500`
- projection_max_token: `500`
- calibration_start_token: `50`
- kv_bytes_per_token_per_sequence: `225280`
- kv_mib_per_token_per_sequence: `0.214844`
- sequences: `1`
- non_moe_mib: `1573.588982`
- avg_expert_mib: `3.757617`
- figures: `experiments/gemma4_IO_behaviors/mem8G/expert_capacity/figures/capacity_estimate_budget_model_decode500_recal_20260519_1150`

## Calibrated Fixed Overhead

| memory | memory limit MiB | fixed overhead MiB |
|---|---:|---:|
| mem8G | 8192.000 | 2166.301 |

## Target Token Error

| memory | observed token | observed median experts | predicted median experts | abs error median | abs error p90 |
|---|---:|---:|---:|---:|---:|
| mem8G | 5 | 942.327 | 1184.537 | 242.210 | 242.210 |
| mem8G | 50 | 1218.711 | 1181.964 | 36.747 | 36.747 |
| mem8G | 100 | 1112.075 | 1179.105 | 67.030 | 67.030 |
| mem8G | 200 | 1219.277 | 1173.388 | 45.889 | 45.889 |
| mem8G | 300 | 1215.220 | 1167.670 | 47.550 | 47.550 |
| mem8G | 400 | 1144.717 | 1161.953 | 17.236 | 17.236 |
| mem8G | 500 | 1125.283 | 1156.235 | 30.952 | 30.952 |

## Max Observed Token

| memory | token | abs error p10 | abs error median | abs error p90 |
|---|---:|---:|---:|---:|
| mem8G | 500 | 30.952 | 30.952 | 30.952 |

## Artifacts

- `capacity_predictions_by_sample.csv`
- `capacity_summary.csv`
- `token_predicted_vs_observed.csv`
- `token_predicted_vs_observed_summary.csv`
- figures disabled for this run

Prediction formula: `(memory_limit - non_MOE - fixed_runtime_overhead - KV(token)) / avg_expert_size`. The fixed runtime overhead is calibrated from observed resident experts at and after `calibration_start_token` unless explicitly provided.
