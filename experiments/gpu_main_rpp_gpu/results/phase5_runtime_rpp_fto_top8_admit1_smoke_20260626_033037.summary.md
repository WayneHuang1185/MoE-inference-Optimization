# Phase 5 Runtime RPP Hint-Admission Summary

- result: `phase5_runtime_rpp_fto_top8_admit1_smoke_20260626_033037.jsonl`
- requests: 2

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | demand MB | demand H2D MB | RPP hint H2D MB | total H2D MB | cache hit % | RPP hint hit % | D2D MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase5-two-step-demand-only-runtime-cache-4096mb | 1 | 1 | 5.684 | 2.878 | 5.630 | 7605 | 5831.5 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25754.0 | 9880.2 | 0.0 | 9880.2 | 61.6 | 0.0 | 25774.7 | 15538.0 | 5119.018 |
| phase5-two-step-real-rpp-fto_admit1_freq64-top8-runtime-cache-4096mb | 1 | 1 | 5.575 | 3.127 | 5.740 | 7244 | 5547.2 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25754.0 | 9794.7 | 180.7 | 9975.4 | 61.9 | 93.0 | 25774.7 | 15404.0 | 4990.876 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
