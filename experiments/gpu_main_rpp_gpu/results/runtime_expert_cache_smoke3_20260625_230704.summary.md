# Phase 1 CPU-MoE Trace Summary

- result: `runtime_expert_cache_smoke3_20260625_230704.jsonl`
- requests: 1

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | H2D payload MB | H2D copied MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-expert-gpu-offload | 1 | 1 | 7.037 | 2.937 | 4.547 | 8472 | 6146.2 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25581.7 | 25602.3 | 40176.0 | 6390.529 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
