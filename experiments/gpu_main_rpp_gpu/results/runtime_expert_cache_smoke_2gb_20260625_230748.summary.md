# Phase 1 CPU-MoE Trace Summary

- result: `runtime_expert_cache_smoke_2gb_20260625_230748.jsonl`
- requests: 1

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | H2D payload MB | H2D copied MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-expert-gpu-offload | 1 | 1 | 5.826 | 2.890 | 5.492 | 7413 | 5953.8 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 14407.7 | 14419.2 | 22634.0 | 5223.143 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
