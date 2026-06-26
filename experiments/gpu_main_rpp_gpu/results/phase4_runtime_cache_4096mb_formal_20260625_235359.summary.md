# Phase 1 CPU-MoE Trace Summary

- result: `phase4_runtime_cache_4096mb_formal_20260625_235359.jsonl`
- requests: 60

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | demand MB | actual H2D MB | copied MB | cache hit % | D2D MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-expert-gpu-offload-runtime-cache-4096mb | 1 | 60 | 7.014 | 3.958 | 4.517 | 9942 | 6990.3 | CPU,CUDA0 | 7920.0 | 3960.0 | 3957.0 | 26213.6 | 12739.4 | 12749.6 | 51.2 | 26234.6 | 20016.5 | 6399.657 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
