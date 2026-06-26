# Offline Real RPP First-Token Analysis

這份結果使用真實 RPP checkpoint，而不是 oracle。流程是先從既有 formal trace 的 generated preview 取回 first generated token，使用 `prompt + first token` 跑 RPP，再只分析 continuation demands。

注意：這仍然是 offline 分析，不會改變原本那次推論 latency；deadline 欄位使用 trace event time 與指定 H2D bandwidth 做估計。

- source result: `/home/hazcashi/lab/experience/RPP-GPU/results/phase2_oracle_ubatch_trace_formal_20260624_171414.jsonl`
- RPP checkpoint: `/home/hazcashi/lab/experience/RPP/qwen36_rpp/results/rpp_train_d64/checkpoint_best.pt`
- requests: 60
- top-k: 2, 4, 8
- top-k 0 in tables means demand-only GPU cache baseline without RPP prefetch.
- cache MB: 0, 1024, 2048, 4096, 6144
- assumed H2D bandwidth: 12 GB/s

## Aggregate Results

| top-k | cache MB | hit rate | H2D miss GB | saved GB | reduction | pred recall(payload) | late pred GB | false positive GB | avg prefetch finish ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0% | 1100.7 | 0.0 | 0.0% | 0.0% | 0.0 | 0.0 | 0.0 |
| 0 | 1024 | 42.4% | 634.4 | 466.3 | 42.4% | 0.0% | 0.0 | 0.0 | 0.0 |
| 0 | 2048 | 56.9% | 475.2 | 625.6 | 56.8% | 0.0% | 0.0 | 0.0 | 0.0 |
| 0 | 4096 | 68.9% | 342.4 | 758.3 | 68.9% | 0.0% | 0.0 | 0.0 | 0.0 |
| 0 | 6144 | 70.1% | 329.4 | 771.3 | 70.1% | 0.0% | 0.0 | 0.0 | 0.0 |
| 2 | 0 | 0.0% | 1100.7 | 0.0 | 0.0% | 4.1% | 0.1 | 2.8 | 18.7 |
| 2 | 1024 | 42.8% | 629.6 | 471.1 | 42.8% | 4.1% | 0.1 | 2.8 | 18.7 |
| 2 | 2048 | 57.3% | 469.9 | 630.9 | 57.3% | 4.1% | 0.1 | 2.8 | 18.7 |
| 2 | 4096 | 69.4% | 336.5 | 764.2 | 69.4% | 4.1% | 0.1 | 2.8 | 18.7 |
| 2 | 6144 | 70.7% | 323.2 | 777.5 | 70.6% | 4.1% | 0.1 | 2.8 | 18.7 |
| 4 | 0 | 0.0% | 1100.7 | 0.0 | 0.0% | 7.5% | 0.2 | 5.3 | 31.5 |
| 4 | 1024 | 43.2% | 625.6 | 475.1 | 43.2% | 7.5% | 0.2 | 5.3 | 31.5 |
| 4 | 2048 | 57.8% | 464.8 | 635.9 | 57.8% | 7.5% | 0.2 | 5.3 | 31.5 |
| 4 | 4096 | 70.0% | 330.4 | 770.4 | 70.0% | 7.5% | 0.2 | 5.3 | 31.5 |
| 4 | 6144 | 71.3% | 316.6 | 784.1 | 71.2% | 7.5% | 0.2 | 5.3 | 31.5 |
| 8 | 0 | 0.0% | 1100.7 | 0.0 | 0.0% | 13.2% | 0.4 | 11.1 | 56.9 |
| 8 | 1024 | 43.7% | 619.8 | 480.9 | 43.7% | 13.2% | 0.4 | 11.1 | 56.9 |
| 8 | 2048 | 58.6% | 455.7 | 645.1 | 58.6% | 13.2% | 0.4 | 11.1 | 56.9 |
| 8 | 4096 | 71.0% | 319.2 | 781.5 | 71.0% | 13.2% | 0.4 | 11.1 | 56.9 |
| 8 | 6144 | 72.4% | 304.3 | 796.5 | 72.4% | 13.2% | 0.4 | 11.1 | 56.9 |

## Request Notes

- tokenizer prefix fallback count: 0
- mean RPP forward time: 5.98 ms
- mean continuation trace duration: 4045.3 ms

## Figures

- [real_rpp_h2d_miss_by_topk_cache.png](../figures/real_rpp_h2d_miss_by_topk_cache.png)
- [real_rpp_cache_hit_rate_by_topk_cache.png](../figures/real_rpp_cache_hit_rate_by_topk_cache.png)
- [real_rpp_late_payload_4gb.png](../figures/real_rpp_late_payload_4gb.png)
- [real_rpp_prediction_recall_4gb.png](../figures/real_rpp_prediction_recall_4gb.png)
- [real_rpp_false_positive_payload_4gb.png](../figures/real_rpp_false_positive_payload_4gb.png)
