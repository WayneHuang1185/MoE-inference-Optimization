# Phase 6 Qwen Online RPP-GPU Formal 0626_1441

## Setup

- 20 prompts x repeat 3 = 60 requests per scenario.
- `n_predict=32`, `ctx_size=512`, `--ngl 999 --cpu-moe`.
- Runtime: `/home/hazcashi/lab/rpp_runtime_implementation/llama.cpp/build-rpp-cuda124/bin/llama-server`.
- Model: `/home/hazcashi/lab/model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- Online RPP sidecar runs on CPU.

## Main Results

| scenario | requests | latency s | tok/s | prediction coverage % | RPP hit % | ready-hit % | total H2D GB/req | sidecar ms |
|---|---|---|---|---|---|---|---|---|
| rpp-off-native | 60 | 6.084 | 10.757 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| demand-cache-1g | 60 | 9.734 | 4.942 | 0.000 | 0.000 | 36.653 | 11.626 | 0.000 |
| online-rpp-top2 | 60 | 11.027 | 4.300 | 100.000 | 8.561 | 36.580 | 16.233 | 4.375 |
| online-rpp-top4 | 60 | 10.768 | 4.264 | 100.000 | 16.202 | 17.211 | 24.134 | 4.419 |
| online-rpp-top8 | 60 | 11.558 | 3.915 | 100.000 | 27.381 | 1.232 | 31.787 | 4.805 |

## Interpretation

- `online-rpp-top2/4/8` all successfully connected real RPP predictions to runtime: prediction coverage is 100%.
- At 1GB cache, online RPP does not improve latency over demand-only cache. The best RPP latency here is top4 at 10.768s/request, still slower than demand-cache at 9.734s/request.
- top-k increases prediction hit rate, but also increases prefetch traffic and cache churn. top8 has the highest RPP hit rate but the worst total H2D payload and lowest throughput.
- The CPU sidecar inference itself is small on average, around 4-5ms/request-sidecar call, so the current bottleneck is mostly prefetch/cache policy and memory movement, not the RPP model forward time.
- This confirms the runtime path is functional, but naive online RPP prefetch is not yet an optimization under this 1GB-cache setting.

## Figures

- `results/figures/phase6_formal_latency.png`
- `results/figures/phase6_formal_throughput.png`
- `results/figures/phase6_formal_total_h2d_per_request.png`
- `results/figures/phase6_formal_hit_rates.png`
- `results/figures/phase6_formal_latency_by_task.png`

## Raw Files

- raw summary: `/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/formal/phase6_qwen_online_rpp_gpu_formal_0626_1441.summary.csv`
- raw requests: `/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/formal/phase6_qwen_online_rpp_gpu_formal_0626_1441.requests.csv`
- compact CSV: `/home/hazcashi/lab/experience/RPP-GPU/results/phase6_qwen_online_rpp_gpu_formal_0626_1441.compact.csv`
- by-task CSV: `/home/hazcashi/lab/experience/RPP-GPU/results/phase6_qwen_online_rpp_gpu_formal_0626_1441.by_task.csv`
