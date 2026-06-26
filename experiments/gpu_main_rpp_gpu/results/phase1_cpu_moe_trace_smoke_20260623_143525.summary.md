# Phase 1 CPU-MoE Trace Summary

- result: `phase1_cpu_moe_trace_smoke_20260623_143525.jsonl`
- requests: 1

| scenario | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | H2D payload MB | H2D copied MB | ranges | enqueue ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-phase1 | 1 | 5.100 | 2.651 | 6.275 | 144317 | 5611.9 | CPU,CUDA0 | 8160.0 | 4080.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.000 |

註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。
