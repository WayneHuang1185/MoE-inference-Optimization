# RPP 100% Oracle vs Baseline

比較對象：

- Baseline：目前 `--ngl 999 --cpu-moe` + on-demand selected-expert H2D copy，沒有 RPP cache。
- RPP 100% oracle：用真實 router selected experts 當 hint，模擬不同 VRAM expert cache 容量。

注意：RPP 100% 目前是離線 upper-bound，因此這份比較只看 H2D payload / cache hit；不把 latency 當成已實測改善。

## Baseline 實測

- requests: 60
- mean total latency: 8.098 s
- mean TTFT: 4.048 s
- mean throughput: 3.866 tok/s
- mean disk read: 7981.2 MB/request
- baseline H2D payload: 1572.8 GB total / 26.21 GB per request

## RPP 100% Oracle Cache 模擬

| cache | hit rate | H2D miss GB | saved GB | reduction | per request miss GB |
|---:|---:|---:|---:|---:|---:|
| baseline | 0.0% | 1572.8 | 0.0 | 0.0% | 26.21 |
| 1 GB | 30.5% | 1093.7 | 479.2 | 30.5% | 18.23 |
| 2 GB | 43.8% | 884.1 | 688.7 | 43.8% | 14.73 |
| 4 GB | 63.3% | 577.7 | 995.1 | 63.3% | 9.63 |
| 6 GB | 75.5% | 385.8 | 1187.0 | 75.5% | 6.43 |

## Figures

- [oracle_vs_baseline_h2d_total_payload.png](../figures/oracle_vs_baseline_h2d_total_payload.png)
- [oracle_vs_baseline_h2d_per_request_payload.png](../figures/oracle_vs_baseline_h2d_per_request_payload.png)
- [oracle_vs_baseline_h2d_reduction.png](../figures/oracle_vs_baseline_h2d_reduction.png)
- [oracle_vs_baseline_cache_hit_rate.png](../figures/oracle_vs_baseline_cache_hit_rate.png)
