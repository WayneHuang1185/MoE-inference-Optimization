# Phase 1 CPU-MoE Trace Summary

- result: `phase4_runtime_cache_2048mb_formal_20260625_233533.jsonl`
- requests: 60

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | demand MB | actual H2D MB | copied MB | cache hit % | D2D MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-expert-gpu-offload-runtime-cache-2048mb | 1 | 60 | 7.308 | 3.929 | 4.316 | 10127 | 7062.7 | CPU,CUDA0 | 7920.0 | 3960.0 | 3957.0 | 26213.6 | 16415.0 | 16428.2 | 37.2 | 26234.6 | 25780.8 | 6724.351 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
