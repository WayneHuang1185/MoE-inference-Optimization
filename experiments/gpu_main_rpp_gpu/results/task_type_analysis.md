# RPP-GPU Task Type Analysis

- source result: `/home/hazcashi/lab/experience/RPP-GPU/results/phase2_oracle_ubatch_trace_formal_20260624_171414.jsonl`
- requests: 60
- summary CSV: `/home/hazcashi/lab/experience/RPP-GPU/results/task_type_analysis.csv`

這份分析將正式 RPP-GPU trace 依 prompt `task_type` 分組。每一類包含 4 個 prompts，每個 prompt repeat 3 次，因此每類共 12 requests。

| task type | requests | total s | TTFT s | tok/s | major faults | read GB/request | H2D GB/request | used experts/request |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text_continuation | 12 | 7.181 | 3.135 | 4.469 | 8574 | 6.43 | 25.45 | 39967 |
| code_generation | 12 | 7.585 | 3.388 | 4.224 | 9108 | 7.13 | 25.98 | 40797 |
| commonsense | 12 | 7.751 | 3.480 | 4.164 | 9208 | 6.77 | 25.67 | 40312 |
| math_reasoning | 12 | 9.191 | 4.968 | 3.485 | 12362 | 9.72 | 28.33 | 44491 |
| multiple_choice | 12 | 8.784 | 5.267 | 2.989 | 11963 | 9.86 | 25.65 | 40290 |

## Figures

- [task_type_total_latency.png](figures/task_type_total_latency.png)
- [task_type_ttft.png](figures/task_type_ttft.png)
- [task_type_throughput.png](figures/task_type_throughput.png)
- [task_type_h2d_payload.png](figures/task_type_h2d_payload.png)
- [task_type_disk_read.png](figures/task_type_disk_read.png)
- [task_type_major_faults.png](figures/task_type_major_faults.png)
- [task_type_used_experts.png](figures/task_type_used_experts.png)
