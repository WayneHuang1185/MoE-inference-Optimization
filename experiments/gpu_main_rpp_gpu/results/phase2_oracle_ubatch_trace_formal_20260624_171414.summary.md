# Phase 1 CPU-MoE Trace Summary

- result: `phase2_oracle_ubatch_trace_formal_20260624_171414.jsonl`
- requests: 60

| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | H2D payload MB | H2D copied MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-expert-gpu-offload | 1 | 60 | 8.098 | 4.048 | 3.866 | 10243 | 7981.2 | CPU,CUDA0 | 7920.0 | 3960.0 | 3957.0 | 26213.6 | 26232.8 | 37612.8 | 7674.992 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
