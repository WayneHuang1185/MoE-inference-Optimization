# GPU-main RPP-GPU 實驗資料

這個資料夾保存 GPU-main 相關的實驗腳本、結果摘要、圖表與 prompt。

## 內容

| 路徑 | 說明 |
|---|---|
| `EXPERIMENT_NOTES.md` | Phase-by-phase 實驗紀錄。 |
| `figures/` | 報告用圖表。 |
| `results/` | compact summary、CSV、Markdown 結果。 |
| `prompts/prompts.jsonl` | 正式實驗使用的 20 prompts。 |
| `run_phase6_qwen_online_rpp_gpu_formal.py` | online RPP-GPU formal runner。 |
| `run_qwen_online_rpp_gpu_smoke.sh` | smoke test runner。 |
| `run_runtime_rpp_hint.py` | runtime RPP hint runner。 |
| `build_oracle_hints.py` | offline oracle hint builder。 |
| `build_real_rpp_offline.py` | offline real-RPP cache simulation builder。 |
| `plot_*.py` | 圖表產生腳本。 |

大型 raw trace、server log 與模型檔案沒有放入 Git。
