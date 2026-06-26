# GPU-main：RPP-GPU Expert Cache / Online Prefetch

這個 branch 是 GPU-main 部分的交付資料，只保留我的 RPP-GPU 實作、實驗紀錄、圖表與報告，不包含其他組員的 CPU-only 或 GPU-CPU mix 內容。

本研究的重點是 Qwen3.6 MoE 在 8GB VRAM 筆電 GPU 上的 expert weight 管理。核心問題是完整 MoE expert weights 無法常駐 GPU VRAM，因此推論時會產生大量 CPU DRAM 到 GPU VRAM 的 host-to-device transfer。本 branch 嘗試用 RPP 預測 upcoming experts，搭配 GPU expert cache 與 runtime prefetch，觀察是否能降低 H2D 傳輸與推論延遲。

## 資料夾結構

```text
.
├── README.md
├── report/
│   ├── REPORT.md
│   └── REPORT_FULL.md
├── experiments/
│   └── gpu_main_rpp_gpu/
│       ├── EXPERIMENT_NOTES.md
│       ├── figures/
│       ├── results/
│       ├── prompts/
│       ├── run_*.py / *.sh
│       └── plot_*.py
└── runtime/
    ├── RPP_GPU_RUNTIME.patch
    └── changed_files/
        └── llama.cpp/
```

## 各資料夾內容

| 路徑 | 說明 |
|---|---|
| `report/REPORT.md` | HackMD-ready 的最終報告版本。 |
| `report/REPORT_FULL.md` | 較完整的工作版報告與中間紀錄。 |
| `experiments/gpu_main_rpp_gpu/` | RPP-GPU 實驗腳本、summary、CSV、圖表與 20 prompts。 |
| `runtime/RPP_GPU_RUNTIME.patch` | 對 llama.cpp 的 RPP-GPU runtime 修改 patch。 |
| `runtime/changed_files/llama.cpp/` | 只保留本研究修改或新增過的 llama.cpp 檔案。 |

## Runtime 實作範圍

RPP-GPU runtime 主要包含：

- GPU expert cache
- host-to-device expert transfer 管理
- RPP prefetch / admission path
- replay predictor / online sidecar interface
- server 端 RPP sidecar integration
- RPP-GPU 相關 unit tests

對應檔案可在以下位置找到：

```text
runtime/changed_files/llama.cpp/src/llama-rpp-*.cpp
runtime/changed_files/llama.cpp/src/llama-rpp-*.h
runtime/changed_files/llama.cpp/tools/server/server-rpp-sidecar.*
runtime/changed_files/llama.cpp/tests/test-rpp-*.cpp
```

若要套用到完整 llama.cpp，可使用：

```bash
cd /path/to/llama.cpp
git apply /path/to/this/repo/runtime/RPP_GPU_RUNTIME.patch
```

## 實驗資料

主要實驗結果整理在：

```text
experiments/gpu_main_rpp_gpu/results/
experiments/gpu_main_rpp_gpu/figures/
```

其中 `results/` 只保留 compact summary、CSV 與 Markdown 結果，沒有包含大型 raw trace。報告中使用的圖片都集中在 `figures/`。

## 未納入 Git 的資料

以下資料刻意不放進 Git：

- GGUF model weights
- raw trace JSONL
- server logs
- build output
- Python virtual environment
- 大型 checkpoint / dataset

這些檔案通常過大、與本機環境綁定，或可由實驗重新產生，因此不適合直接提交到 GitHub。

## 目前結論

目前 RPP-GPU pipeline 已經能接通 online RPP sidecar、runtime GPU expert cache 與 H2D 統計；但正式實驗中，未篩選的 top-k RPP prefetch 尚未勝過 native GPU baseline。比較有價值的觀察是：GPU expert cache 在 component level 能有效降低重複 H2D payload，下一步應該往 batch-aware / ubatch-aware admission、FTO policy、async prefetch 與 compute-copy overlap 前進。
