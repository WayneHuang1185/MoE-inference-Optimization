# Phase 5 Runtime RPP Hint-Admission Summary

- result: `phase5_runtime_rpp_hint_top2_smoke_20260626_023233.jsonl`
- requests: 2

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | demand MB | demand H2D MB | RPP hint H2D MB | total H2D MB | cache hit % | RPP hint hit % | D2D MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase5-two-step-demand-only-runtime-cache-4096mb | 1 | 1 | 5.852 | 3.109 | 5.468 | 7880 | 5903.2 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25754.0 | 9880.2 | 0.0 | 9880.2 | 61.6 | 0.0 | 25774.7 | 15538.0 | 5280.259 |
| phase5-two-step-real-rpp-top2-runtime-cache-4096mb | 1 | 1 | 5.377 | 2.858 | 5.951 | 7551 | 5197.7 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25754.0 | 9862.1 | 366.6 | 10228.6 | 61.7 | 92.9 | 25774.7 | 15509.0 | 4808.487 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
