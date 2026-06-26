# Phase 5 Runtime RPP Hint-Admission Summary

- result: `phase5_runtime_rpp_hint_smoke_20260626_023112.jsonl`
- requests: 2

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | demand MB | demand H2D MB | RPP hint H2D MB | total H2D MB | cache hit % | RPP hint hit % | D2D MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase5-two-step-demand-only-runtime-cache-4096mb | 1 | 1 | 5.417 | 3.038 | 5.907 | 7520 | 5662.3 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25754.0 | 9880.2 | 0.0 | 9880.2 | 61.6 | 0.0 | 25774.7 | 15538.0 | 4843.070 |
| phase5-two-step-real-rpp-top8-runtime-cache-4096mb | 1 | 1 | 6.030 | 3.456 | 5.307 | 7540 | 5589.5 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25754.0 | 10146.3 | 1531.8 | 11678.2 | 60.5 | 92.6 | 25774.7 | 15956.0 | 5289.358 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
