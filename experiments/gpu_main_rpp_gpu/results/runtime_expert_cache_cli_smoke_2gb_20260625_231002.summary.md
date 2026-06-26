# Phase 1 CPU-MoE Trace Summary

- result: `runtime_expert_cache_cli_smoke_2gb_20260625_231002.jsonl`
- requests: 1

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | demand MB | actual H2D MB | copied MB | cache hit % | D2D MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-expert-gpu-offload-runtime-cache-2048mb | 1 | 1 | 6.079 | 2.935 | 5.264 | 8265 | 6007.4 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25581.7 | 14407.7 | 14419.2 | 43.7 | 25602.3 | 22634.0 | 5507.636 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
