# Final Runtime Comparison 0626

## Fresh Baseline Rerun

| scenario | requests | latency s | tok/s |
|---|---:|---:|---:|
| native baseline rerun | 60 | 6.023 | 11.164 |

## Phase 6 Formal With Fresh Baseline

| scenario | latency s | tok/s | RPP hit | ready-hit | total H2D GB/request |
|---|---:|---:|---:|---:|---:|
| rpp-off-native | 6.023 | 11.164 | 0.0% | 0.0% | 0.000 |
| demand-cache-1g | 9.734 | 4.942 | 0.0% | 36.7% | 11.626 |
| online-rpp-top2 | 11.027 | 4.300 | 8.6% | 36.6% | 16.233 |
| online-rpp-top4 | 10.768 | 4.264 | 16.2% | 17.2% | 24.134 |
| online-rpp-top8 | 11.558 | 3.915 | 27.4% | 1.2% | 31.787 |

## Interpretation

- No RPP-GPU formal scenario beats the fresh native baseline. The best RPP-GPU formal latency is demand-cache-1g at 9.734s/request, while the fresh native baseline is 6.023s/request.
- Within RPP-GPU scenarios, demand-only cache remains better than online RPP top2/top4/top8. Online RPP has 100% prediction coverage, but naive top-k prefetch increases total H2D and cache churn.
- Historical Phase 4 demand-only cache still shows that GPU expert cache can reduce H2D payload, but this is a different implementation stage and should be read as a component result rather than a new best baseline.
- Phase 5 hint smoke suggests FTO/admission policy is more promising than naive top-k, but it was one-prompt smoke and still does not establish a formal improvement over baseline.

## Figures

- `results/figures/final_phase6_latency_vs_baseline.png`
- `results/figures/final_phase6_throughput_vs_baseline.png`
- `results/figures/final_phase6_total_h2d_vs_baseline.png`
- `results/figures/final_phase6_hit_rate_vs_ready_hit.png`
- `results/figures/final_phase4_cache_latency_vs_baseline.png`
- `results/figures/final_phase4_cache_h2d_reduction.png`
- `results/figures/final_phase5_hint_latency_delta.png`
- `results/figures/final_phase5_hint_h2d_delta.png`

## Raw Inputs

- `/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/formal/phase6_qwen_native_baseline_rerun_0626_1537.summary.csv`
- `/home/hazcashi/lab/experience/RPP-GPU/results/phase6_qwen_online_rpp_gpu_formal_0626_1441.compact.csv`
- `/home/hazcashi/lab/experience/RPP-GPU/results/phase4_runtime_cache_formal_comparison.csv`
- `/home/hazcashi/lab/experience/RPP-GPU/results/phase5_runtime_rpp_hint_smoke_comparison.csv`
