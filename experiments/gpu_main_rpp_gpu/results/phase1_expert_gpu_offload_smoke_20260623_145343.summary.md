# Phase 1 CPU-MoE Trace Summary

- result: `phase1_expert_gpu_offload_smoke_20260623_145343.jsonl`
- requests: 1

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | H2D payload MB | H2D copied MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-expert-gpu-offload | 1 | 1 | 6.478 | 2.708 | 4.940 | 8116 | 5739.2 | CPU,CUDA0 | 8160.0 | 4080.0 | 4077.0 | 25581.7 | 25600.9 | 37569.0 | 6089.423 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
