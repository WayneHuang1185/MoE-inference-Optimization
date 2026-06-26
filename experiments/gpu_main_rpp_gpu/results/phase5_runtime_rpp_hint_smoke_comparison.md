# Phase 5 Runtime RPP Hint-Admission Smoke

這份結果測的是 runtime RPP hint admission：RPP hint 會進入 C++ GPU expert cache，但還沒有 background async prefetch / compute-copy overlap。

- summary CSV: `phase5_runtime_rpp_hint_smoke_comparison.csv`

| variant | RPP ms | hint slots | demand-only s | RPP hint s | latency delta | demand-only total H2D MB | RPP demand H2D MB | RPP hint H2D MB | RPP total H2D MB | total H2D delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FTO top-8/admit-1 | 1148.2 | 40 | 5.684 | 5.575 | -0.109 | 9880.2 | 9794.7 | 180.7 | 9975.4 | +95.2 |
| FTO top-8/admit-2 | 1271.0 | 79 | 5.745 | 6.065 | +0.321 | 9880.2 | 9741.3 | 358.3 | 10099.6 | +219.3 |
| Rank top-2 | 1103.6 | 80 | 5.852 | 5.377 | -0.474 | 9880.2 | 9862.1 | 366.6 | 10228.6 | +348.4 |
| Rank top-4 | 1232.7 | 160 | 5.321 | 6.225 | +0.904 | 9880.2 | 10013.2 | 740.8 | 10753.9 | +873.7 |
| Rank top-8 | 1102.3 | 320 | 5.417 | 6.030 | +0.613 | 9880.2 | 10146.3 | 1531.8 | 11678.2 | +1798.0 |

## Figures

- [phase5_runtime_rpp_hint_total_h2d.png](figures/phase5_runtime_rpp_hint_total_h2d.png)
- [phase5_runtime_rpp_hint_latency.png](figures/phase5_runtime_rpp_hint_latency.png)
- [phase5_runtime_rpp_hint_extra_h2d.png](figures/phase5_runtime_rpp_hint_extra_h2d.png)
