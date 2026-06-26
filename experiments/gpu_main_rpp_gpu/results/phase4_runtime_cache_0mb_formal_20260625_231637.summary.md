# Phase 1 CPU-MoE Trace Summary

- result: `phase4_runtime_cache_0mb_formal_20260625_231637.jsonl`
- requests: 60

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | demand MB | actual H2D MB | copied MB | cache hit % | D2D MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-expert-gpu-offload | 1 | 60 | 7.753 | 3.856 | 4.026 | 9732 | 7194.5 | CPU,CUDA0 | 7920.0 | 3960.0 | 3957.0 | 26213.6 | 26213.6 | 26232.8 | 0.0 | 0.0 | 37612.8 | 7323.574 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
