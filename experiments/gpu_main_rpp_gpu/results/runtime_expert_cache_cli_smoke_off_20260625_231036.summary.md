# Phase 1 CPU-MoE Trace Summary

- result: `runtime_expert_cache_cli_smoke_off_20260625_231036.jsonl`
- requests: 1

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | demand MB | actual H2D MB | copied MB | cache hit % | D2D MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-expert-gpu-offload | 1 | 1 | 6.663 | 2.992 | 4.802 | 7571 | 6093.9 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25581.7 | 25581.7 | 25600.9 | 0.0 | 0.0 | 37569.0 | 6285.564 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
