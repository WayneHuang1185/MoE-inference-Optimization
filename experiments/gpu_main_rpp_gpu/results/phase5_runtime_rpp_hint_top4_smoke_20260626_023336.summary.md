# Phase 5 Runtime RPP Hint-Admission Summary

- result: `phase5_runtime_rpp_hint_top4_smoke_20260626_023336.jsonl`
- requests: 2

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | demand MB | demand H2D MB | RPP hint H2D MB | total H2D MB | cache hit % | RPP hint hit % | D2D MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase5-two-step-demand-only-runtime-cache-4096mb | 1 | 1 | 5.321 | 2.909 | 6.014 | 6852 | 5511.7 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25754.0 | 9880.2 | 0.0 | 9880.2 | 61.6 | 0.0 | 25774.7 | 15538.0 | 4843.932 |
| phase5-two-step-real-rpp-top4-runtime-cache-4096mb | 1 | 1 | 6.225 | 3.605 | 5.141 | 8683 | 5200.6 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25754.0 | 10013.2 | 740.8 | 10753.9 | 61.1 | 92.9 | 25774.7 | 15746.0 | 5436.753 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
