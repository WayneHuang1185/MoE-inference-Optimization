# Offline Real RPP First-Token Analysis

這份結果使用真實 RPP checkpoint，而不是 oracle。流程是先從既有 formal trace 的 generated preview 取回 first generated token，使用 `prompt + first token` 跑 RPP，再只分析 continuation demands。

注意：這仍然是 offline 分析，不會改變原本那次推論 latency；deadline 欄位使用 trace event time 與指定 H2D bandwidth 做估計。

- source result: `/home/hazcashi/lab/experience/RPP-GPU/results/phase2_oracle_ubatch_trace_formal_20260624_171414.jsonl`
- RPP checkpoint: `/home/hazcashi/lab/experience/RPP/qwen36_rpp/results/rpp_train_d64/checkpoint_best.pt`
- requests: 1
- top-k: 2, 4, 8
- top-k 0 in tables means demand-only GPU cache baseline without RPP prefetch.
- cache MB: 0, 1024, 2048, 4096, 6144
- assumed H2D bandwidth: 12 GB/s

## Aggregate Results

| top-k | cache MB | hit rate | H2D miss GB | saved GB | reduction | pred recall(payload) | late pred GB | false positive GB | avg prefetch finish ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0% | 19.0 | 0.0 | 0.0% | 0.0% | 0.0 | 0.0 | 0.0 |
| 0 | 1024 | 49.4% | 9.6 | 9.4 | 49.4% | 0.0% | 0.0 | 0.0 | 0.0 |
| 0 | 2048 | 61.9% | 7.2 | 11.7 | 61.9% | 0.0% | 0.0 | 0.0 | 0.0 |
| 0 | 4096 | 75.5% | 4.6 | 14.3 | 75.5% | 0.0% | 0.0 | 0.0 | 0.0 |
| 0 | 6144 | 75.7% | 4.6 | 14.4 | 75.7% | 0.0% | 0.0 | 0.0 | 0.0 |
| 2 | 0 | 0.0% | 19.0 | 0.0 | 0.0% | 5.4% | 0.0 | 0.1 | 165.7 |
| 2 | 1024 | 49.4% | 9.6 | 9.4 | 49.4% | 5.4% | 0.0 | 0.1 | 165.7 |
| 2 | 2048 | 62.0% | 7.2 | 11.8 | 62.0% | 5.4% | 0.0 | 0.1 | 165.7 |
| 2 | 4096 | 75.7% | 4.6 | 14.4 | 75.7% | 5.4% | 0.0 | 0.1 | 165.7 |
| 2 | 6144 | 75.9% | 4.6 | 14.4 | 76.0% | 5.4% | 0.0 | 0.1 | 165.7 |
| 4 | 0 | 0.0% | 19.0 | 0.0 | 0.0% | 11.0% | 0.1 | 0.1 | 178.5 |
| 4 | 1024 | 49.2% | 9.6 | 9.3 | 49.2% | 11.0% | 0.1 | 0.1 | 178.5 |
| 4 | 2048 | 62.2% | 7.2 | 11.8 | 62.2% | 11.0% | 0.1 | 0.1 | 178.5 |
| 4 | 4096 | 76.1% | 4.5 | 14.4 | 76.1% | 11.0% | 0.1 | 0.1 | 178.5 |
| 4 | 6144 | 76.3% | 4.5 | 14.5 | 76.3% | 11.0% | 0.1 | 0.1 | 178.5 |
| 8 | 0 | 0.0% | 19.0 | 0.0 | 0.0% | 16.8% | 0.1 | 0.2 | 203.9 |
| 8 | 1024 | 48.5% | 9.8 | 9.2 | 48.5% | 16.8% | 0.1 | 0.2 | 203.9 |
| 8 | 2048 | 62.4% | 7.1 | 11.8 | 62.4% | 16.8% | 0.1 | 0.2 | 203.9 |
| 8 | 4096 | 76.7% | 4.4 | 14.5 | 76.7% | 16.8% | 0.1 | 0.2 | 203.9 |
| 8 | 6144 | 77.0% | 4.4 | 14.6 | 77.0% | 16.8% | 0.1 | 0.2 | 203.9 |

## Request Notes

- tokenizer prefix fallback count: 0
- mean RPP forward time: 152.99 ms
- mean continuation trace duration: 3846.4 ms

## Figures

- [real_rpp_h2d_miss_by_topk_cache.png](../figures/real_rpp_h2d_miss_by_topk_cache.png)
- [real_rpp_cache_hit_rate_by_topk_cache.png](../figures/real_rpp_cache_hit_rate_by_topk_cache.png)
- [real_rpp_late_payload_4gb.png](../figures/real_rpp_late_payload_4gb.png)
- [real_rpp_prediction_recall_4gb.png](../figures/real_rpp_prediction_recall_4gb.png)
- [real_rpp_false_positive_payload_4gb.png](../figures/real_rpp_false_positive_payload_4gb.png)
