# Offline Oracle RPP Hints

這份結果把 llama.cpp router 實際選到的 experts 當成 RPP 預測，因此 prediction accuracy 等於 100%。
注意：這是離線 upper-bound 分析，還沒有把 hint 接進 runtime prefetch 或 VRAM cache。

## 輸入與輸出

- hint JSONL: `offline_oracle_rpp_smoke_0624_1638.hints.jsonl`
- cache simulation CSV: `offline_oracle_rpp_smoke_0624_1638.cache.csv`
- traces: 1
- hints / selected-expert copy events: 4077
- true ubatches from trace: 0
- inferred ubatches from old trace: 34

## 目前 on-demand copy 基準

- H2D payload: 25581.7 MB
- H2D copied bytes with padding: 25600.9 MB
- enqueue time sum: 6089.4 ms
- infinite-cache lower bound payload: 6943.9 MB
- perfect-cache maximum payload saved: 18637.8 MB

## GPU Expert Cache 模擬

| cache MB | hit rate | demand MB | miss MB | saved MB | hits | misses |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 25581.7 | 25581.7 | 0.0 | 0 | 40176 |
| 256 | 0.000 | 25581.7 | 25581.7 | 0.0 | 0 | 40176 |
| 512 | 0.000 | 25581.7 | 25581.7 | 0.0 | 0 | 40176 |
| 1024 | 0.373 | 25581.7 | 16049.4 | 9532.4 | 14971 | 25205 |

解讀：`miss MB` 是 perfect predictor 仍然必須搬進 VRAM cache 的 expert payload。
如果 cache 很小且 hit rate 很低，RPP 的主要價值只剩 prefetch/overlap；如果 cache 能帶來明顯 hit，RPP+cache 才可能同時減少 H2D bytes。

## Trace 明細

| trace | hints | true ubatches | inferred ubatches | selected ids |
|---|---:|---:|---:|---:|
| `phase1_expert_gpu_offload_smoke_20260623_145343_text_001_r1.trace.jsonl` | 4077 | 0 | 34 | 40176 |
