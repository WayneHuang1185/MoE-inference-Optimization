# Phase 4 Runtime GPU Expert Cache Formal Comparison

這份比較使用 20 個 prompts、repeat 3，總共 60 requests。所有結果都是真實 runtime demand-only GPU expert cache；尚未接入 RPP async prefetch。

- `demand MB/request`：如果沒有 cache，每個 request selected experts 需要搬的 H2D payload。
- `actual H2D MB/request`：runtime cache 後實際從 CPU host memory 搬到 GPU 的 payload。
- `D2D MB/request`：目前 hit/miss 後仍需 staging 到原本 compute buffer 的 GPU-to-GPU copy，不是 CPU/GPU 傳輸。

- overall CSV: `phase4_runtime_cache_formal_comparison.csv`
- task CSV: `phase4_runtime_cache_formal_by_task.csv`

## Overall

| cache | requests | total s | TTFT s | tok/s | demand MB/request | actual H2D MB/request | H2D reduction | cache hit | D2D MB/request | major faults | read MB/request |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0MB | 60 | 7.753 | 3.856 | 4.026 | 26213.6 | 26213.6 | 0.0% | 0.0% | 0.0 | 9732 | 7194.5 |
| 2GB | 60 | 7.308 | 3.929 | 4.316 | 26213.6 | 16415.0 | 37.4% | 37.2% | 26234.6 | 10127 | 7062.7 |
| 4GB | 60 | 7.014 | 3.958 | 4.517 | 26213.6 | 12739.4 | 51.4% | 51.2% | 26234.6 | 9942 | 6990.3 |

## By Task Type

| task | cache | total s | tok/s | actual H2D MB/request | H2D reduction | cache hit | major faults |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text | 0MB | 7.052 | 4.553 | 25448.5 | 0.0% | 0.0% | 8497 |
| Text | 2GB | 6.426 | 5.010 | 15035.9 | 40.9% | 40.9% | 8176 |
| Text | 4GB | 5.813 | 5.529 | 10702.7 | 57.9% | 58.0% | 8152 |
| Code | 0MB | 7.505 | 4.292 | 25977.3 | 0.0% | 0.0% | 8792 |
| Code | 2GB | 6.635 | 4.845 | 15493.7 | 40.4% | 40.4% | 8746 |
| Code | 4GB | 6.583 | 4.876 | 11616.4 | 55.3% | 55.3% | 8892 |
| Commonsense | 0MB | 7.308 | 4.392 | 25668.3 | 0.0% | 0.0% | 8937 |
| Commonsense | 2GB | 7.033 | 4.578 | 15713.9 | 38.8% | 38.8% | 9543 |
| Commonsense | 4GB | 6.519 | 4.928 | 11728.3 | 54.3% | 54.3% | 9224 |
| Math | 0MB | 8.779 | 3.655 | 28325.2 | 0.0% | 0.0% | 11405 |
| Math | 2GB | 8.251 | 3.902 | 18862.1 | 33.4% | 33.4% | 11932 |
| Math | 4GB | 8.480 | 3.805 | 15507.5 | 45.3% | 45.2% | 11770 |
| MC | 0MB | 8.123 | 3.236 | 25648.7 | 0.0% | 0.0% | 11028 |
| MC | 2GB | 8.193 | 3.246 | 16969.5 | 33.8% | 32.6% | 12238 |
| MC | 4GB | 7.673 | 3.448 | 14142.1 | 44.9% | 43.1% | 11670 |

## Figures

- [phase4_runtime_cache_actual_h2d.png](figures/phase4_runtime_cache_actual_h2d.png)
- [phase4_runtime_cache_hit_rate.png](figures/phase4_runtime_cache_hit_rate.png)
- [phase4_runtime_cache_latency.png](figures/phase4_runtime_cache_latency.png)
- [phase4_runtime_cache_throughput.png](figures/phase4_runtime_cache_throughput.png)
- [phase4_runtime_cache_d2d.png](figures/phase4_runtime_cache_d2d.png)
- [phase4_runtime_cache_task_h2d.png](figures/phase4_runtime_cache_task_h2d.png)
- [phase4_runtime_cache_task_latency.png](figures/phase4_runtime_cache_task_latency.png)
- [phase4_runtime_cache_task_hit_rate.png](figures/phase4_runtime_cache_task_hit_rate.png)
