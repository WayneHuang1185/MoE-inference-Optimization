# RPP Prefill Generalization

## Run

- statistics: `experiments/gemma4_global_predictor/statistics/rpp_prefill_generalization_20260624_023216`
- samples: `1000`
- checkpoint_epoch: `60`
- wall_s: `141.945`
- topks: `16`

## Phase Counts

| phase | valid tokens | valid layer tokens |
|---|---:|---:|
| prefill | 108656.000000 | 3259680.000000 |
| decode | 9888.000000 | 296640.000000 |
| all_valid | 118544.000000 | 3556320.000000 |

## Prefill vs Decode

| metric | prefill | decode | prefill - decode | prefill / decode |
|---|---:|---:|---:|---:|
| token_recall@16 | 0.429168 | 0.884846 | -0.455678 | 0.485020 |
| token_precision@16 | 0.208325 | 0.430822 | -0.222496 | 0.483554 |
| token_top1 | 0.134449 | 0.389499 | -0.255050 | 0.345184 |
| batch_level_accuracy@16 | 0.883528 | 0.957841 | -0.074314 | 0.922415 |
| kl_true_pred | 8.449227 | 6.856969 | 1.592258 | 1.232210 |

## Decode Layer Top-k Recall

`figures/decode_layer_precision_20260520_161202/decode_layer_topk_precision_heatmap.svg` summarizes decode token positions with layer-wise top-k target recall. For each token $t$ and layer $l$, let $G_{t,l}$ denote the ground-truth expert set selected by the router, with $|G_{t,l}|=8$, and let $\hat{P}^{(k)}_{t,l}$ denote the top-$k$ RPP candidate expert set. Top-k recall is defined as $|\hat{P}^{(k)}_{t,l} \cap G_{t,l}| / \min(k, |G_{t,l}|)$.

| decode top-k | mean recall | expected correct experts |
|---|---:|---:|
| top2 | 0.916 | 1.83 |
| top4 | 0.870 | 3.48 |
| top6 | 0.820 | 4.92 |
| top8 | 0.753 | 6.02 |
| top16 | 0.895 | 7.16 |

## Artifacts

- `phase_metrics.csv`
- `layer_phase_metrics.csv`
- `metric_comparison.csv`
- `summary.json`

## Output Format

- `statistics/`: remote CSV, JSON, and Markdown reports.
- `figures/`: experiment data images only; this run does not generate figures.
- `utils/`: reserved for reusable experiment tools.
