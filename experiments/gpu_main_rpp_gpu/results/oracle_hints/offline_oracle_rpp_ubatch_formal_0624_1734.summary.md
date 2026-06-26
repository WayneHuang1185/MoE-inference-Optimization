# Offline Oracle RPP Hints

這份結果把 llama.cpp router 實際選到的 experts 當成 RPP 預測，因此 prediction accuracy 等於 100%。
注意：這是離線 upper-bound 分析，還沒有把 hint 接進 runtime prefetch 或 VRAM cache。

## 輸入與輸出

- hint JSONL: `offline_oracle_rpp_ubatch_formal_0624_1734.hints.jsonl`
- cache simulation CSV: `offline_oracle_rpp_ubatch_formal_0624_1734.cache.csv`
- traces: 60
- hints / selected-expert copy events: 237420
- true ubatches from trace: 1980
- inferred ubatches from old trace: 0

## 目前 on-demand copy 基準

- H2D payload: 1572816.2 MB
- H2D copied bytes with padding: 1573967.5 MB
- enqueue time sum: 460499.5 ms
- infinite-cache lower bound payload: 17855.0 MB
- perfect-cache maximum payload saved: 1554961.2 MB

## GPU Expert Cache 模擬

| cache MB | hit rate | demand MB | miss MB | saved MB | hits | misses |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1572816.2 | 1572816.2 | 0.0 | 0 | 2470275 |
| 256 | 0.000 | 1572816.2 | 1572816.2 | 0.0 | 0 | 2470275 |
| 512 | 0.000 | 1572816.2 | 1572816.2 | 0.0 | 0 | 2470275 |
| 1024 | 0.305 | 1572816.2 | 1093658.4 | 479157.8 | 752679 | 1717596 |
| 2048 | 0.438 | 1572816.2 | 884084.2 | 688732.0 | 1081551 | 1388724 |
| 4096 | 0.633 | 1572816.2 | 577675.0 | 995141.2 | 1563088 | 907187 |
| 6144 | 0.755 | 1572816.2 | 385793.9 | 1187022.3 | 1864754 | 605521 |

解讀：`miss MB` 是 perfect predictor 仍然必須搬進 VRAM cache 的 expert payload。
如果 cache 很小且 hit rate 很低，RPP 的主要價值只剩 prefetch/overlap；如果 cache 能帶來明顯 hit，RPP+cache 才可能同時減少 H2D bytes。

## Trace 明細

| trace | hints | true ubatches | inferred ubatches | selected ids |
|---|---:|---:|---:|---:|
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_001_r1.trace.jsonl` | 4077 | 34 | 0 | 40176 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_002_r1.trace.jsonl` | 4077 | 34 | 0 | 40104 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_003_r1.trace.jsonl` | 4077 | 34 | 0 | 40554 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_004_r1.trace.jsonl` | 4077 | 34 | 0 | 39033 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_001_r1.trace.jsonl` | 4077 | 34 | 0 | 44463 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_002_r1.trace.jsonl` | 4077 | 34 | 0 | 44829 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_003_r1.trace.jsonl` | 4077 | 34 | 0 | 45015 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_004_r1.trace.jsonl` | 4077 | 34 | 0 | 43656 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_001_r1.trace.jsonl` | 4077 | 34 | 0 | 40188 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_002_r1.trace.jsonl` | 4077 | 34 | 0 | 41844 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_003_r1.trace.jsonl` | 4077 | 34 | 0 | 40446 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_004_r1.trace.jsonl` | 4077 | 34 | 0 | 40710 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_001_r1.trace.jsonl` | 4077 | 34 | 0 | 43965 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_002_r1.trace.jsonl` | 4077 | 34 | 0 | 45078 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_003_r1.trace.jsonl` | 4077 | 34 | 0 | 46065 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_004_r1.trace.jsonl` | 1677 | 14 | 0 | 26052 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_001_r1.trace.jsonl` | 4077 | 34 | 0 | 40974 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_002_r1.trace.jsonl` | 4077 | 34 | 0 | 39345 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_003_r1.trace.jsonl` | 4077 | 34 | 0 | 39855 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_004_r1.trace.jsonl` | 4077 | 34 | 0 | 41073 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_001_r2.trace.jsonl` | 4077 | 34 | 0 | 40176 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_002_r2.trace.jsonl` | 4077 | 34 | 0 | 40104 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_003_r2.trace.jsonl` | 4077 | 34 | 0 | 40554 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_004_r2.trace.jsonl` | 4077 | 34 | 0 | 39033 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_001_r2.trace.jsonl` | 4077 | 34 | 0 | 44463 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_002_r2.trace.jsonl` | 4077 | 34 | 0 | 44829 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_003_r2.trace.jsonl` | 4077 | 34 | 0 | 45015 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_004_r2.trace.jsonl` | 4077 | 34 | 0 | 43656 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_001_r2.trace.jsonl` | 4077 | 34 | 0 | 40188 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_002_r2.trace.jsonl` | 4077 | 34 | 0 | 41844 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_003_r2.trace.jsonl` | 4077 | 34 | 0 | 40446 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_004_r2.trace.jsonl` | 4077 | 34 | 0 | 40710 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_001_r2.trace.jsonl` | 4077 | 34 | 0 | 43965 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_002_r2.trace.jsonl` | 4077 | 34 | 0 | 45078 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_003_r2.trace.jsonl` | 4077 | 34 | 0 | 46065 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_004_r2.trace.jsonl` | 1677 | 14 | 0 | 26052 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_001_r2.trace.jsonl` | 4077 | 34 | 0 | 40974 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_002_r2.trace.jsonl` | 4077 | 34 | 0 | 39345 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_003_r2.trace.jsonl` | 4077 | 34 | 0 | 39855 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_004_r2.trace.jsonl` | 4077 | 34 | 0 | 41073 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_001_r3.trace.jsonl` | 4077 | 34 | 0 | 40176 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_002_r3.trace.jsonl` | 4077 | 34 | 0 | 40104 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_003_r3.trace.jsonl` | 4077 | 34 | 0 | 40554 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_text_004_r3.trace.jsonl` | 4077 | 34 | 0 | 39033 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_001_r3.trace.jsonl` | 4077 | 34 | 0 | 44463 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_002_r3.trace.jsonl` | 4077 | 34 | 0 | 44829 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_003_r3.trace.jsonl` | 4077 | 34 | 0 | 45015 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_math_004_r3.trace.jsonl` | 4077 | 34 | 0 | 43656 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_001_r3.trace.jsonl` | 4077 | 34 | 0 | 40188 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_002_r3.trace.jsonl` | 4077 | 34 | 0 | 41844 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_003_r3.trace.jsonl` | 4077 | 34 | 0 | 40446 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_code_004_r3.trace.jsonl` | 4077 | 34 | 0 | 40710 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_001_r3.trace.jsonl` | 4077 | 34 | 0 | 43965 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_002_r3.trace.jsonl` | 4077 | 34 | 0 | 45078 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_003_r3.trace.jsonl` | 4077 | 34 | 0 | 46065 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_mc_004_r3.trace.jsonl` | 1677 | 14 | 0 | 26052 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_001_r3.trace.jsonl` | 4077 | 34 | 0 | 40974 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_002_r3.trace.jsonl` | 4077 | 34 | 0 | 39345 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_003_r3.trace.jsonl` | 4077 | 34 | 0 | 39855 |
| `phase2_oracle_ubatch_trace_formal_20260624_171414_common_004_r3.trace.jsonl` | 4077 | 34 | 0 | 41073 |
