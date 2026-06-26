# RPP Analyze Result Log

這份文件集中整理 `experience/RPP/analyze/` 底下所有本機 IO 實驗結果。
每次新增 baseline、GPU、RPP prefetch、MTP 或 oracle-window 實驗後，都把
參數、跑了什麼、結果摘要、圖表連結和觀察補在這裡。

## 圖表整理原則

- latency、page faults、throughput 分開畫，不全部擠在同一張圖。
- 每張圖只回答一個問題，例如「TTFT 是否下降」、「major faults 是否下降」。
- 同一類實驗使用一致 scenario name，方便跨階段比較。
- 圖表檔名使用內容命名，不放完整 timestamp，例如 `gpu_all_baseline_throughput.png`。
- 報告中的實驗日期只記錄月日，例如 `06/19`。
- 原始資料保留 JSONL，摘要保留 CSV/Markdown，圖表放在 `figures/`。

## Experiment 001（06/18）：CPU Baseline Cold/Warm

### 實驗環境

```text
OS:
  Ubuntu Linux，kernel 7.0.0-14-generic，x86_64

Machine:
  OMEN by HP Gaming Laptop 16-xf0xxx

Memory:
  RAM 約 14GiB
  swap 約 4GiB，/swap.img

Swap 狀態（實驗後檢查）:
  SwapTotal 約 4,194,300 kB
  SwapFree 約 2,448,200 kB
  Swap used 約 1.7GiB

GPU:
  RTX 4060 Laptop GPU
  本次 CPU baseline 明確使用 --device none / --no-op-offload，不使用 GPU。
```

### 目的

建立 CPU-only、沒有 RPP prefetch 的 baseline，作為後續 GPU baseline、
RPP first-token prefetch、oracle future-window、MTP 整合的比較基準。

這次實驗回答：

```text
在本機 14GiB RAM、21G Qwen3.6 GGUF、CPU-only mmap 條件下，
單純 warm workload 是否能降低 latency 或 page faults？
```

### 執行內容

執行兩個 scenario：

```text
baseline-cpu-cold
baseline-cpu-warm
```

定義：

```text
baseline-cpu-cold:
  對 model file 執行 posix_fadvise(DONTNEED)，啟動 llama-server，
  然後連續跑固定 prompt set。

baseline-cpu-warm:
  不重啟 server，不重新 evict model pages，直接用同一個 server
  再跑同一批 prompts。
```

注意：這裡的 cold/warm 是「workload 層級」比較，不是每一個 prompt 都重新
啟動 fresh server 的 per-request cold start。

### 參數

```text
model:
  /home/hazcashi/lab/model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

llama-server:
  /home/hazcashi/lab/llama.cpp/build/bin/llama-server

RPP checkpoint:
  ../qwen36_rpp/results/rpp_train_d64/checkpoint_best.pt
  本次 baseline 尚未使用 RPP，只先記錄後續會使用的 checkpoint。

prompts:
  prompts.jsonl
  20 prompts，包含 text_continuation、math_reasoning、code_generation、
  multiple_choice、commonsense，各 4 條。

generation:
  n_predict = 32
  temperature = 0.0
  ctx = 2048
  threads = 8
  repeat = 3

CPU-only:
  -ngl 0
  --device none
  --no-op-offload
  --mmap
  --no-warmup
```

### 資料檔案

```text
raw:
  local_io_20260618_194958.jsonl

summary:
  local_io_20260618_194958.summary.md
  local_io_20260618_194958.summary.csv

server log:
  logs/baseline_cpu_cold_server.log
```

### 圖表

Latency、faults、throughput 分開成多張圖，避免單張圖資訊過載：

```text
figures/cpu_baseline_ttft.png
figures/cpu_baseline_total_latency.png
figures/cpu_baseline_throughput.png
figures/cpu_baseline_major_faults.png
figures/cpu_baseline_minor_faults.png
figures/cpu_baseline_total_latency_by_task.png
```

### 結果摘要

| scenario | requests | prompts | mean TTFT (s) | mean total (s) | mean tok/s | mean major faults | mean minor faults |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline-cpu-cold | 60 | 20 | 5.385 | 32.393 | 0.962 | 624,187 | 1,807,404 |
| baseline-cpu-warm | 60 | 20 | 5.472 | 32.398 | 0.959 | 625,713 | 1,829,696 |

### 初步觀察

- warm workload 幾乎沒有改善 latency。
- warm workload 的 major page faults 也沒有下降，甚至平均略高。
- 這代表在目前本機條件下，模型頁面無法有效保留在 RAM/page cache。
- 目前機器 RAM 約 14GiB，而 GGUF 約 21G，CPU-only mmap 推理會持續受到
  page faults / memory pressure 影響。
- 本次 runner 尚未記錄每個 request 前後的 SwapFree / SwapCached delta；
  已在後續版本加入，下一輪實驗會同步觀測 swap 狀態。
- 這是一個適合測 RPP IO prefetch 的 baseline，因為 bottleneck 明顯存在。

### 後續比較方向

- 加入 GPU baseline，確認 GPU offload 能否降低 latency 和 page faults。
- 加入 RPP-predicted expert prefetch，比較 major faults 是否下降。
- 加入 oracle future-window，估計 MTP 提供 future tokens 後的上限。
- 後續圖表需將 baseline、GPU、RPP 分階段疊加比較。

## Experiment 002（06/19）：GPU Baseline / 8GiB VRAM 可行設定

### 目的

確認 RTX 4060 Laptop GPU 啟用後，在 8GiB VRAM 限制下，哪些 llama.cpp
GPU offload 設定可以穩定跑完整 workload，並建立後續 RPP GPU 實驗的 baseline。

### GPU 啟用狀態

```text
nvidia-smi:
  driver 595.71.05
  CUDA 13.2
  RTX 4060 Laptop GPU
  VRAM 8188MiB

llama-server --list-devices:
  CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU
```

### 參數

```text
model:
  /home/hazcashi/lab/model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

llama-server:
  /home/hazcashi/lab/llama.cpp/build/bin/llama-server

prompts:
  prompts.jsonl
  20 prompts，repeat = 3，總共 60 requests / scenario

generation:
  n_predict = 32
  temperature = 0.0
  ctx = 2048
  threads = 8

shared:
  --mmap
  --no-warmup
```

### 2A. GPU Smoke：找 VRAM 上限

資料檔案：

```text
raw:
  local_io_20260619_113726.jsonl

summary:
  local_io_20260619_113726.summary.md
  local_io_20260619_113726.summary.csv
```

Smoke 只跑 1 prompt、repeat 1，用來找可行設定，不拿來當正式吞吐比較。

| scenario | 結果 | mean TTFT (s) | mean total (s) | mean tok/s | mean major faults | 備註 |
|---|---|---:|---:|---:|---:|---|
| baseline-gpu-ngl5 | success | 2.521 | 5.550 | 5.766 | 120,536 | offloaded 5/41 layers |
| baseline-gpu-ngl10 | success | 2.585 | 5.732 | 5.583 | 151,425 | offloaded 10/41 layers |
| baseline-gpu-ngl15 | failed | - | - | - | - | CUDA compute buffer OOM，server returncode -11 |
| baseline-gpu-ngl20 | failed | - | - | - | - | CUDA model buffer OOM，server returncode 1 |
| baseline-gpu-cpu-moe | success | 2.195 | 4.141 | 7.728 | 107,395 | `-ngl 999 --cpu-moe` |

這裡的 OOM 是 `Out Of Memory`，代表記憶體不夠。這次不是 RAM 滿，而是 GPU
VRAM 不夠，所以 CUDA 在配置 model buffer 或 compute buffer 時失敗。

Smoke 圖表：

```text
figures/gpu_smoke_ttft.png
figures/gpu_smoke_total_latency.png
figures/gpu_smoke_throughput.png
figures/gpu_smoke_major_faults.png
figures/gpu_smoke_minor_faults.png
figures/gpu_smoke_swap_free_delta.png
figures/gpu_smoke_swap_cached_delta.png
figures/gpu_smoke_total_latency_by_task.png
```

重要 log 摘要：

```text
baseline-gpu-ngl10:
  offloaded 10/41 layers to GPU
  CPU_Mapped model buffer = 16085.26 MiB
  CUDA0 model buffer = 5013.39 MiB

baseline-gpu-ngl15:
  CUDA0 model buffer = 7525.58 MiB
  但 compute buffer 553.01 MiB 配不下，cudaMalloc failed: out of memory

baseline-gpu-ngl20:
  嘗試配置 CUDA0 model buffer 10037.76 MiB
  超過 8GiB VRAM，cudaMalloc failed: out of memory

baseline-gpu-cpu-moe:
  offloaded 41/41 layers to GPU
  CPU_Mapped model buffer = 20699.72 MiB
  CUDA0 model buffer = 1921.34 MiB
```

### 2B. 正式 GPU Baseline

正式跑兩個可行 scenario：

```text
baseline-gpu-ngl10:
  -ngl 10
  一般 partial layer offload。

baseline-gpu-cpu-moe:
  -ngl 999
  --cpu-moe
  讓可 offload 的部分使用 GPU，但 MoE expert 權重主要留在 CPU mapped memory。
```

資料檔案：

```text
raw:
  local_io_20260619_113910.jsonl

summary:
  local_io_20260619_113910.summary.md
  local_io_20260619_113910.summary.csv

server logs:
  logs/local_io_20260619_113910_baseline_gpu_ngl10_server.log
  logs/local_io_20260619_113910_baseline_gpu_cpu_moe_server.log
```

結果摘要：

| scenario | requests | prompts | mean TTFT (s) | mean total (s) | mean tok/s | mean major faults | mean minor faults | mean SwapFree delta KB | mean SwapCached delta KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-gpu-cpu-moe | 60 | 20 | 1.416 | 2.865 | 12.041 | 17,782 | 64,176 | -15,633 | -49 |
| baseline-gpu-ngl10 | 60 | 20 | 1.262 | 3.105 | 10.179 | 10,612 | 46,363 | -21,760 | -668 |

相對 CPU cold baseline：

| scenario | total latency speedup | TTFT speedup | tok/s speedup | major faults reduction |
|---|---:|---:|---:|---:|
| baseline-gpu-ngl10 | 10.43x | 4.27x | 10.58x | 98.30% |
| baseline-gpu-cpu-moe | 11.31x | 3.80x | 12.52x | 97.15% |

### 圖表

GPU formal 各指標分開畫：

```text
figures/gpu_offload_baseline_ttft.png
figures/gpu_offload_baseline_total_latency.png
figures/gpu_offload_baseline_throughput.png
figures/gpu_offload_baseline_major_faults.png
figures/gpu_offload_baseline_minor_faults.png
figures/gpu_offload_baseline_swap_free_delta.png
figures/gpu_offload_baseline_swap_cached_delta.png
figures/gpu_offload_baseline_total_latency_by_task.png
```

CPU baseline + GPU baseline 跨實驗比較：

```text
figures/cpu_vs_gpu_baseline_ttft.png
figures/cpu_vs_gpu_baseline_total_latency.png
figures/cpu_vs_gpu_baseline_throughput.png
figures/cpu_vs_gpu_baseline_major_faults.png
figures/cpu_vs_gpu_baseline_minor_faults.png
figures/cpu_vs_gpu_baseline_swap_free_delta.png
figures/cpu_vs_gpu_baseline_swap_cached_delta.png
figures/cpu_vs_gpu_baseline_total_latency_by_task.png
```

### 初步觀察

- 8GiB VRAM 無法承受 `--ngl 15` 以上的一般 layer offload；`--ngl 20`
  會直接需要約 10GiB CUDA model buffer。
- `baseline-gpu-ngl10` 的 TTFT 較低、major faults 較少，但平均 tok/s 低於
  `baseline-gpu-cpu-moe`。
- `baseline-gpu-cpu-moe` 的 throughput 和 total latency 最好，但 major
  faults 高於 `ngl10`，表示它仍保留大量 CPU mapped model buffer。
- 對 RPP IO prefetch 來說，`baseline-gpu-cpu-moe` 是很有價值的主 baseline：
  它已經有 GPU 加速，但 expert 權重仍主要受 CPU DRAM / page cache / SSD IO
  影響。
- 後續 RPP GPU 實驗建議優先對照 `baseline-gpu-cpu-moe`，另外保留
  `baseline-gpu-ngl10` 作為 partial offload 參考組。

## Experiment 003（06/19）：GPU Utilization / Partial MoE Offload Smoke

### 目的

回答 `--ngl 999 --cpu-moe` 是否能吃滿 GPU，以及能不能透過 partial MoE
offload 提高 GPU 使用率。

這不是正式 20 prompts baseline，只是 1 prompt、repeat 1 的調參 smoke。

### 背景

```text
--cpu-moe:
  所有 MoE expert weights 留在 CPU。

--n-cpu-moe N:
  前 N 層 MoE expert weights 留在 CPU，其餘 MoE expert weights 嘗試放進 GPU。
  N 越小，GPU VRAM 使用越高，也越容易 OOM。
```

### 資料檔案

```text
raw:
  local_io_20260619_120550.jsonl

summary:
  local_io_20260619_120550.summary.md
  local_io_20260619_120550.summary.csv

server logs:
  logs/local_io_20260619_120550_baseline_gpu_ncpu_moe_40_server.log
  logs/local_io_20260619_120550_baseline_gpu_ncpu_moe_36_server.log
  logs/local_io_20260619_120550_baseline_gpu_ncpu_moe_32_server.log
  logs/local_io_20260619_120550_baseline_gpu_ncpu_moe_28_server.log
```

### 結果摘要

| scenario | 結果 | mean TTFT (s) | mean total (s) | mean tok/s | mean major faults | CUDA model buffer |
|---|---|---:|---:|---:|---:|---:|
| baseline-gpu-ncpu-moe-40 | success | 2.408 | 4.586 | 6.977 | 163,123 | 1.9GiB |
| baseline-gpu-ncpu-moe-36 | success | 2.337 | 4.368 | 7.326 | 135,663 | 3.8GiB |
| baseline-gpu-ncpu-moe-32 | success | 2.258 | 4.169 | 7.676 | 113,897 | 5.7GiB |
| baseline-gpu-ncpu-moe-28 | failed | - | - | - | - | 7.6GiB + context buffer 後 OOM |

額外取樣：

```text
baseline-gpu-cpu-moe:
  VRAM 約 2.8GiB
  GPU utilization 取樣大多 0%-24%

baseline-gpu-ncpu-moe-32:
  VRAM 峰值約 6.6GiB
  GPU utilization 取樣峰值約 69%
```

### 圖表

```text
figures/gpu_ncpu_moe_smoke_ttft.png
figures/gpu_ncpu_moe_smoke_total_latency.png
figures/gpu_ncpu_moe_smoke_throughput.png
figures/gpu_ncpu_moe_smoke_major_faults.png
figures/gpu_ncpu_moe_smoke_minor_faults.png
figures/gpu_ncpu_moe_smoke_swap_free_delta.png
figures/gpu_ncpu_moe_smoke_swap_cached_delta.png
figures/gpu_ncpu_moe_smoke_total_latency_by_task.png
```

### 初步觀察

- `--ngl 999 --cpu-moe` 沒有吃滿 GPU，因為 MoE experts 留在 CPU 側。
- `--n-cpu-moe 32` 可以把 VRAM 使用拉到約 6.6GiB，GPU utilization 峰值也
  明顯高於 `--cpu-moe`。
- `--n-cpu-moe 28` 已經太激進，雖然 model buffer 約 7.6GiB，但後續 context /
  recurrent state / compute buffer 配不下，所以 OOM。
- 提高 GPU utilization 不等於一定提高單 request latency。單 request decode
  仍可能被 CPU、DRAM、page faults 或 CPU-GPU 搬移卡住。
- 後續若想測「更吃 GPU 的 RPP baseline」，可以正式跑
  `baseline-gpu-ncpu-moe-32`，再和 `baseline-gpu-cpu-moe` 比較。

## Experiment 004（06/19）：正式 GPU MoE Baseline

### 目的

正式比較兩個 8GiB VRAM 下可行的 GPU MoE baseline：

```text
baseline-gpu-cpu-moe:
  -ngl 999 --cpu-moe
  所有 MoE expert weights 留在 CPU 側。

baseline-gpu-ncpu-moe-32:
  -ngl 999 --n-cpu-moe 32
  前 32 層 MoE experts 留在 CPU，其餘 MoE experts 嘗試 offload 到 GPU。
```

這次要回答：

```text
提高 GPU utilization 之後，total latency / throughput / page faults
是否真的比 --cpu-moe baseline 更好？
```

### 資料檔案

```text
raw:
  local_io_20260619_121257.jsonl

summary:
  local_io_20260619_121257.summary.md
  local_io_20260619_121257.summary.csv

server logs:
  logs/local_io_20260619_121257_baseline_gpu_cpu_moe_server.log
  logs/local_io_20260619_121257_baseline_gpu_ncpu_moe_32_server.log
```

### 參數

```text
prompts:
  prompts.jsonl
  20 prompts，repeat = 3，總共 60 requests / scenario

generation:
  n_predict = 32
  temperature = 0.0
  ctx = 2048
  threads = 8

shared:
  --mmap
  --no-warmup
```

### 結果摘要

| scenario | requests | prompts | mean TTFT (s) | mean total (s) | mean tok/s | mean major faults | mean minor faults | mean SwapFree delta KB | mean SwapCached delta KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-gpu-cpu-moe | 60 | 20 | 1.356 | 2.736 | 12.383 | 16,373 | 63,784 | -21,082 | -326 |
| baseline-gpu-ncpu-moe-32 | 60 | 20 | 0.882 | 1.892 | 17.464 | 8,952 | 40,969 | -20,204 | -761 |

相對 `baseline-gpu-cpu-moe`：

| scenario | total latency reduction | TTFT reduction | tok/s increase | major faults reduction | minor faults reduction |
|---|---:|---:|---:|---:|---:|
| baseline-gpu-ncpu-moe-32 | 30.85% | 34.96% | 41.03% | 45.32% | 35.77% |

### Server Log 摘要

```text
baseline-gpu-cpu-moe:
  CPU_Mapped model buffer = 20699.72 MiB
  CUDA0 model buffer = 1921.34 MiB
  CUDA0 compute buffer = 501.03 MiB

baseline-gpu-ncpu-moe-32:
  CPU_Mapped model buffer = 16581.03 MiB
  CUDA0 model buffer = 5735.34 MiB
  CUDA0 compute buffer = 501.00 MiB
```

### 圖表

```text
figures/gpu_moe_baseline_ttft.png
figures/gpu_moe_baseline_total_latency.png
figures/gpu_moe_baseline_throughput.png
figures/gpu_moe_baseline_major_faults.png
figures/gpu_moe_baseline_minor_faults.png
figures/gpu_moe_baseline_swap_free_delta.png
figures/gpu_moe_baseline_swap_cached_delta.png
figures/gpu_moe_baseline_total_latency_by_task.png
```

### 全部 GPU Baseline 對照

因為 `baseline-gpu-ngl10` 是 Experiment 002 跑的，而
`baseline-gpu-cpu-moe` / `baseline-gpu-ncpu-moe-32` 是 Experiment 004 跑的，
所以另外建立一份 selected JSONL 專門產生三者比較圖。

```text
selected raw:
  gpu_all_baselines_selected.jsonl

selected summary:
  gpu_all_baselines_selected.summary.md
  gpu_all_baselines_selected.summary.csv
```

三者比較摘要：

| scenario | requests | prompts | mean TTFT (s) | mean total (s) | mean tok/s | mean major faults | mean minor faults |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline-gpu-cpu-moe | 60 | 20 | 1.356 | 2.736 | 12.383 | 16,373 | 63,784 |
| baseline-gpu-ncpu-moe-32 | 60 | 20 | 0.882 | 1.892 | 17.464 | 8,952 | 40,969 |
| baseline-gpu-ngl10 | 60 | 20 | 1.262 | 3.105 | 10.179 | 10,612 | 46,363 |

全部 GPU baseline 圖表：

```text
figures/gpu_all_baseline_ttft.png
figures/gpu_all_baseline_total_latency.png
figures/gpu_all_baseline_throughput.png
figures/gpu_all_baseline_major_faults.png
figures/gpu_all_baseline_minor_faults.png
figures/gpu_all_baseline_swap_free_delta.png
figures/gpu_all_baseline_swap_cached_delta.png
figures/gpu_all_baseline_total_latency_by_task.png
```

### 初步觀察

- `baseline-gpu-ncpu-moe-32` 是目前正式 baseline 裡最快的設定。
- 相比 `--cpu-moe`，`--n-cpu-moe 32` 把 CUDA model buffer 從約 1.9GiB
  拉高到約 5.7GiB，代表更多 MoE expert weights 真的進 GPU。
- `--n-cpu-moe 32` 同時降低 latency、提高 throughput，且 major faults
  下降約 45%，表示把部分 expert weights 移到 GPU 後，CPU-side mmap/page
  fault 壓力也明顯降低。
- 後續 RPP 實驗建議保留兩個 baseline：
  `baseline-gpu-cpu-moe` 代表 IO 壓力較大的 setting；
  `baseline-gpu-ncpu-moe-32` 代表目前效能最佳、GPU 使用率較高的 setting。

## Experiment 005（06/19）：Phase 2 RPP First-Token Smoke

### 目的

確認訓練好的 RPP checkpoint 能在本機 Phase 2 流程中實際使用，並先跑
CPU / GPU 兩個 first-token prefetch smoke。

這次不是正式 20 prompts baseline，只是 1 prompt、repeat 1，用來確認：

```text
1. RPP checkpoint 能用 config.json 正確重建並載入。
2. llama.cpp tokenizer 產生的 token ids 能餵給 RPP。
3. RPP logits shape 和 GGUF expert ranges 對得上。
4. RPP predicted slots 可以轉成 byte ranges 並 prefetch。
5. CPU / GPU server 都能完成 first-token + RPP + prefetch + continuation。
```

### RPP Model Check

```text
checkpoint:
  ../qwen36_rpp/results/rpp_train_d64/checkpoint_best.pt

config:
  vocab_size = 249093
  layers = 40
  experts = 256
  d_model = 64

check result:
  prompt_id = text_001
  token_count = 21
  RPP logits shape = [1, 21, 40, 256]
  predicted slots = 320
  slots found in GGUF = 320
  GGUF expert slots = 10240
  GGUF layers = 0..39
  GGUF experts = 0..255
```

注意：這個 checkpoint 必須用同資料夾的 `config.json` 建模型，不能用
`model.py` 預設參數，因為訓練時推得的 vocab size 是 249093。

### Smoke 設定

```text
script:
  run_phase2_rpp_smoke.py

raw:
  phase2_rpp_smoke_20260619_130606.jsonl

summary:
  phase2_rpp_smoke_20260619_130606.summary.md
  phase2_rpp_smoke_20260619_130606.summary.csv

RPP:
  checkpoint_best.pt
  device = cpu
  top_k = 8
  token_window = 1

CPU scenario:
  rpp-first-token-cpu
  -ngl 0 --device none --no-op-offload
  eligible layers = 40

GPU scenario:
  rpp-first-token-gpu
  -ngl 999 --n-cpu-moe 32
  eligible layers = 32
```

GPU 版只 prefetch layer 0..31，因為 `--n-cpu-moe 32` 代表後面 layers 的
MoE experts 已經盡量 offload 到 GPU，讀那些 ranges 對 CPU-side IO 幫助較小。

### 結果摘要

| scenario | requests | prompts | first TTFT (s) | end-to-end total (s) | tok/s | major faults | prefetch bytes | RPP forward (s) | prefetch (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-cpu | 1 | 1 | 3.803 | 14.534 | 2.202 | 200,383 | 612MB | 0.798 | 0.718 |
| rpp-first-token-gpu | 1 | 1 | 2.284 | 5.632 | 5.681 | 116,703 | 487MB | 0.160 | 0.187 |

Continuation page faults：

```text
rpp-first-token-cpu:
  first_token_major_faults = 78,777
  continuation_major_faults = 121,606

rpp-first-token-gpu:
  first_token_major_faults = 90,779
  continuation_major_faults = 25,924
```

### 圖表

```text
figures/phase2_rpp_smoke_ttft.png
figures/phase2_rpp_smoke_total_latency.png
figures/phase2_rpp_smoke_throughput.png
figures/phase2_rpp_smoke_major_faults.png
figures/phase2_rpp_smoke_minor_faults.png
figures/phase2_rpp_smoke_swap_free_delta.png
figures/phase2_rpp_smoke_swap_cached_delta.png
figures/phase2_rpp_smoke_total_latency_by_task.png
```

### 初步觀察

- RPP checkpoint 可用，且 RPP output shape / GGUF expert slot mapping 完全對齊。
- GPU smoke 成功跑通，server log 確認仍是 `--n-cpu-moe 32` 的配置：
  CPU_Mapped model buffer 約 16.6GiB，CUDA0 model buffer 約 5.7GiB。
- RPP CPU forward 在這次 smoke 中約 0.16s 到 0.80s；正式實驗要把 RPP overhead
  和 prefetch time 分開看，不能只看 end-to-end total。
- 這個 Python-side smoke 會用 `prompt + first_token` 發第二次 completion
  request，仍不是 llama.cpp 內部真正 asynchronous continuation prefetch。
因此目前結論是「流程已跑通」，正式效能結論要等 20 prompts / repeat 3。

## Experiment 006（06/19）：正式 Phase 2 RPP First-Token Prefetch

### 目的

正式跑 Phase 2：先生成第一個 token，再用 `prompt + first_token` 餵給 RPP，
把預測出的 MoE expert ranges 做 page-cache prefetch，最後繼續生成剩餘
tokens。這次同時跑 CPU 與目前最快的 GPU baseline 對應設定。

這次主要回答：

```text
RPP first-token prefetch 流程在 20 prompts / repeat 3 下是否穩定？
CPU / GPU 版本的 latency、throughput、page faults 和 prefetch overhead 分別如何？
```

### 設定

```text
script:
  run_phase2_rpp_smoke.py

phase label:
  phase2-rpp-first-token-formal

prompts:
  prompts.jsonl
  20 prompts，repeat = 3，總共 60 requests / scenario

generation:
  n_predict = 32
  temperature = 0.0
  ctx = 2048
  threads = 8

RPP:
  checkpoint_best.pt
  device = cpu
  top_k = 8
  token_window = 1

CPU scenario:
  rpp-first-token-cpu
  -ngl 0 --device none --no-op-offload
  eligible layers = 40

GPU scenario:
  rpp-first-token-gpu
  -ngl 999 --n-cpu-moe 32
  eligible layers = 32
```

注意：GPU 版只 prefetch layer 0..31 的 expert ranges，對齊
`--n-cpu-moe 32` 的 CPU-side MoE layers。

### 資料檔案

```text
raw:
  phase2_rpp_formal_20260619_131451.jsonl

summary:
  phase2_rpp_formal_20260619_131451.summary.md
  phase2_rpp_formal_20260619_131451.summary.csv

phase2 breakdown:
  phase2_rpp_formal_20260619_131451.phase2.md
  phase2_rpp_formal_20260619_131451.phase2.csv
```

### 整體結果摘要

| scenario | requests | prompts | mean TTFT (s) | mean total (s) | mean tok/s | mean major faults | mean minor faults | mean SwapFree delta KB | mean SwapCached delta KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-cpu | 60 | 20 | 5.433 | 16.164 | 1.936 | 210,662 | 649,397 | -10,458 | -3,418 |
| rpp-first-token-gpu | 60 | 20 | 2.991 | 5.828 | 5.489 | 105,538 | 154,873 | -75,106 | 4,403 |

### Phase 2 階段拆解

| scenario | total (s) | first token (s) | continuation (s) | tokenize (ms) | RPP (ms) | prefetch (ms) | prefetch MB | first major | continuation major |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-cpu | 16.164 | 5.436 | 8.956 | 872.1 | 181.8 | 718.3 | 612 | 84,018 | 126,645 |
| rpp-first-token-gpu | 5.828 | 2.994 | 1.752 | 787.0 | 147.4 | 148.4 | 487 | 88,850 | 16,688 |

### 圖表

一般指標：

```text
figures/phase2_rpp_formal_ttft.png
figures/phase2_rpp_formal_total_latency.png
figures/phase2_rpp_formal_throughput.png
figures/phase2_rpp_formal_major_faults.png
figures/phase2_rpp_formal_minor_faults.png
figures/phase2_rpp_formal_swap_free_delta.png
figures/phase2_rpp_formal_swap_cached_delta.png
figures/phase2_rpp_formal_total_latency_by_task.png
```

Phase 2 階段拆解：

```text
figures/phase2_rpp_formal_stage_latency.png
figures/phase2_rpp_formal_major_fault_breakdown.png
figures/phase2_rpp_formal_prefetch_cost.png
```

### 初步觀察

- RPP checkpoint 在正式 120 requests 中可穩定跑完，tokenizer、RPP forward、
  GGUF expert range mapping、prefetch、continuation 都能接起來。
- Phase 2 GPU 版相對 CPU 版 total latency 下降約 63.9%，throughput 提升約
  183.5%，mean major faults 下降約 49.9%。
- GPU 版 continuation major faults 平均約 16.7k，CPU 版約 126.6k；這表示
  `--n-cpu-moe 32` 把部分 MoE 放進 GPU 後，continuation 階段的 CPU-side IO
  壓力明顯低很多。
- RPP forward 本身平均約 0.15s 到 0.18s，還算小；但 Python-side tokenize
  平均約 0.8s，已經是目前流程中不可忽略的 overhead。
- CPU 版平均 prefetch 約 612MB、花 0.718s；GPU 版平均 prefetch 約 487MB、
  花 0.148s。後續若要更公平比較 RPP gain，需要把 no-RPP first-token
  control 也做成同樣的 per-request cold / two-step 流程。

### 限制

這次 Phase 2 還是 Python-side 兩段式流程：第一段 request 只拿 first token，
第二段 request 用 `prompt + first_token` 繼續生成，並靠 server prompt cache
盡量重用前綴。它還不是 llama.cpp 內部真正 asynchronous prefetch，所以不能
直接拿這次 total latency 和早先 server-reuse baseline 做一比一結論。

## Experiment 007（06/19）：GPU Matched Two-Step No-RPP Control

### 目的

補上與 `rpp-first-token-gpu` 相同 two-step 流程、但不做 RPP / prefetch 的
matched control。這組才適合用來判斷目前 Python-side RPP prefetch 本身是否
有 end-to-end gain。

流程：

```text
two-step-no-rpp-gpu:
  prompt -> first token
  prompt + first_token -> 不跑 RPP、不 prefetch
  prompt + first_token -> continuation

rpp-first-token-gpu:
  prompt -> first token
  prompt + first_token -> tokenizer + RPP
  RPP predicted experts -> prefetch
  prompt + first_token -> continuation
```

兩組 GPU 設定相同：

```text
-ngl 999
--n-cpu-moe 32
eligible layers = 32
```

### 資料檔案

```text
no-RPP raw:
  phase2_two_step_no_rpp_gpu_formal_20260619_154954.jsonl

no-RPP summary:
  phase2_two_step_no_rpp_gpu_formal_20260619_154954.summary.md
  phase2_two_step_no_rpp_gpu_formal_20260619_154954.summary.csv
  phase2_two_step_no_rpp_gpu_formal_20260619_154954.phase2.md
  phase2_two_step_no_rpp_gpu_formal_20260619_154954.phase2.csv

matched comparison:
  phase2_gpu_rpp_vs_no_rpp_selected.jsonl
  phase2_gpu_rpp_vs_no_rpp_selected.summary.md
  phase2_gpu_rpp_vs_no_rpp_selected.summary.csv
  phase2_gpu_rpp_vs_no_rpp_selected.phase2.md
  phase2_gpu_rpp_vs_no_rpp_selected.phase2.csv
```

### Matched Comparison 結果

| scenario | requests | prompts | mean total (s) | mean tok/s | mean major faults | first token (s) | continuation (s) | continuation major |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| two-step-no-rpp-gpu | 60 | 20 | 4.991 | 6.424 | 122,608 | 3.060 | 1.931 | 23,009 |
| rpp-first-token-gpu | 60 | 20 | 5.828 | 5.489 | 105,538 | 2.994 | 1.752 | 16,688 |

RPP 相對 no-RPP：

```text
total latency:
  +0.838s，慢 16.78%

throughput:
  -14.55%

mean major faults:
  -13.92%

continuation latency:
  -0.179s，快 9.27%

continuation major faults:
  -27.47%

extra Python-side overhead:
  tokenize + RPP + prefetch = 1.083s / request
```

### 圖表

```text
figures/phase2_gpu_rpp_vs_no_rpp_ttft.png
figures/phase2_gpu_rpp_vs_no_rpp_total_latency.png
figures/phase2_gpu_rpp_vs_no_rpp_throughput.png
figures/phase2_gpu_rpp_vs_no_rpp_major_faults.png
figures/phase2_gpu_rpp_vs_no_rpp_minor_faults.png
figures/phase2_gpu_rpp_vs_no_rpp_swap_free_delta.png
figures/phase2_gpu_rpp_vs_no_rpp_swap_cached_delta.png
figures/phase2_gpu_rpp_vs_no_rpp_total_latency_by_task.png
figures/phase2_gpu_rpp_vs_no_rpp_stage_latency.png
figures/phase2_gpu_rpp_vs_no_rpp_major_fault_breakdown.png
figures/phase2_gpu_rpp_vs_no_rpp_prefetch_cost.png
```

### 初步觀察

- RPP prefetch 有降低 IO 指標：mean major faults 下降約 13.9%，continuation
  major faults 下降約 27.5%。
- RPP 也讓 continuation latency 從 1.931s 降到 1.752s，約快 9.3%。
- 但目前 Python-side overhead 約 1.083s/request，大於 continuation 省下的
  約 0.179s，因此 end-to-end total latency 反而慢約 16.8%。
- 目前瓶頸不是 RPP forward 本身，而是 Python-side tokenizer 約 0.787s，加上
  prefetch 約 0.148s 和 RPP 約 0.147s。
- 若要讓 RPP 在 GPU two-step 下端到端變快，下一步應優先降低 tokenizer /
  Python boundary overhead，或改成 llama.cpp 內部 async prefetch，讓 prefetch
  和 decode overlap。

## Experiment 008（06/19）：CPU Matched Two-Step No-RPP Control

### 目的

補上與 `rpp-first-token-cpu` 相同 two-step 流程、但不做 RPP / prefetch 的
CPU matched control。這組用來判斷在 CPU-only、IO 壓力最大的情境下，RPP
prefetch 是否能轉成端到端收益。

兩組 CPU 設定相同：

```text
-ngl 0
--device none
--no-op-offload
```

### 資料檔案

```text
no-RPP raw:
  phase2_two_step_no_rpp_cpu_formal_20260619_165712.jsonl

no-RPP summary:
  phase2_two_step_no_rpp_cpu_formal_20260619_165712.summary.md
  phase2_two_step_no_rpp_cpu_formal_20260619_165712.summary.csv
  phase2_two_step_no_rpp_cpu_formal_20260619_165712.phase2.md
  phase2_two_step_no_rpp_cpu_formal_20260619_165712.phase2.csv

matched comparison:
  phase2_cpu_rpp_vs_no_rpp_selected.jsonl
  phase2_cpu_rpp_vs_no_rpp_selected.summary.md
  phase2_cpu_rpp_vs_no_rpp_selected.summary.csv
  phase2_cpu_rpp_vs_no_rpp_selected.phase2.md
  phase2_cpu_rpp_vs_no_rpp_selected.phase2.csv
```

### Matched Comparison 結果

| scenario | requests | prompts | mean total (s) | mean tok/s | mean major faults | first token (s) | continuation (s) | continuation major |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| two-step-no-rpp-cpu | 60 | 20 | 12.666 | 2.499 | 175,989 | 5.454 | 7.212 | 89,151 |
| rpp-first-token-cpu | 60 | 20 | 16.164 | 1.936 | 210,662 | 5.436 | 8.956 | 126,645 |

RPP 相對 no-RPP：

```text
total latency:
  +3.498s，慢 27.62%

throughput:
  -22.54%

mean major faults:
  +19.70%

continuation latency:
  +1.744s，慢 24.19%

continuation major faults:
  +42.06%

extra Python-side overhead:
  tokenize + RPP + prefetch = 1.772s / request
```

### 圖表

```text
figures/phase2_cpu_rpp_vs_no_rpp_ttft.png
figures/phase2_cpu_rpp_vs_no_rpp_total_latency.png
figures/phase2_cpu_rpp_vs_no_rpp_throughput.png
figures/phase2_cpu_rpp_vs_no_rpp_major_faults.png
figures/phase2_cpu_rpp_vs_no_rpp_minor_faults.png
figures/phase2_cpu_rpp_vs_no_rpp_swap_free_delta.png
figures/phase2_cpu_rpp_vs_no_rpp_swap_cached_delta.png
figures/phase2_cpu_rpp_vs_no_rpp_total_latency_by_task.png
figures/phase2_cpu_rpp_vs_no_rpp_stage_latency.png
figures/phase2_cpu_rpp_vs_no_rpp_major_fault_breakdown.png
figures/phase2_cpu_rpp_vs_no_rpp_prefetch_cost.png
```

### 初步觀察

- CPU-only 情境下，RPP first-token prefetch 沒有改善端到端表現，反而讓 total
  latency 慢約 27.6%，throughput 下降約 22.5%。
- 和 GPU matched control 不同，CPU 版 RPP 不只 overhead 較高，IO 指標也變差：
  mean major faults 增加約 19.7%，continuation major faults 增加約 42.1%。
- CPU 版平均 prefetch 約 612MB/request，花約 0.718s。這個同步 prefetch 本身
  可能造成額外 memory pressure，讓後續 continuation page faults 變多。
- 目前結論更明確：Python-side RPP prefetch 在 CPU 和 GPU matched control
  都沒有端到端加速；GPU 版有降低 continuation IO，但被 overhead 抵消，CPU
  版則連 IO 指標也變差。

## Experiment 009（06/19）：GPU RPP Top-K Sweep

### 目的

測試 GPU `rpp-first-token-gpu` 在不同 RPP top-k 下的 prefetch tradeoff。
這組回答：

```text
降低 top_k 是否可以減少 over-prefetch / prefetch overhead，
並讓 RPP prefetch 轉成 end-to-end 加速？
```

固定設定：

```text
scenario:
  rpp-first-token-gpu

GPU:
  -ngl 999
  --n-cpu-moe 32

RPP:
  token_window = 1
  rpp_device = cpu

prompts:
  20 prompts，repeat = 3，總共 60 requests / top_k
```

### 資料檔案

```text
top_k = 2:
  phase2_rpp_gpu_topk2_formal_20260619_182024.jsonl
  phase2_rpp_gpu_topk2_formal_20260619_182024.summary.md
  phase2_rpp_gpu_topk2_formal_20260619_182024.phase2.md

top_k = 4:
  phase2_rpp_gpu_topk4_formal_20260619_185214.jsonl
  phase2_rpp_gpu_topk4_formal_20260619_185214.summary.md
  phase2_rpp_gpu_topk4_formal_20260619_185214.phase2.md

top_k = 8:
  phase2_rpp_formal_20260619_131451.jsonl
  phase2_rpp_formal_20260619_131451.summary.md
  phase2_rpp_formal_20260619_131451.phase2.md

combined sweep:
  phase2_gpu_topk_sweep_selected.jsonl
  phase2_gpu_topk_sweep_selected.summary.md
  phase2_gpu_topk_sweep_selected.phase2.md
```

### 結果摘要

| scenario | total (s) | tok/s | major faults | continuation (s) | continuation major | overhead (s) | prefetch MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| two-step-no-rpp-gpu | 4.991 | 6.424 | 122,608 | 1.931 | 23,009 | 0.000 | 0 |
| rpp-first-token-gpu-topk2 | 5.933 | 5.313 | 88,135 | 2.021 | 25,735 | 0.997 | 122 |
| rpp-first-token-gpu-topk4 | 6.029 | 5.295 | 73,776 | 1.984 | 23,794 | 1.058 | 243 |
| rpp-first-token-gpu-topk8 | 5.828 | 5.489 | 105,538 | 1.752 | 16,688 | 1.083 | 487 |

相對 no-RPP：

```text
top_k = 2:
  total latency +18.88%
  major faults -28.12%
  continuation latency +4.67%
  continuation major faults +11.85%

top_k = 4:
  total latency +20.82%
  major faults -39.83%
  continuation latency +2.77%
  continuation major faults +3.41%

top_k = 8:
  total latency +16.78%
  major faults -13.92%
  continuation latency -9.27%
  continuation major faults -27.47%
```

### 圖表

```text
figures/phase2_gpu_topk_sweep_ttft.png
figures/phase2_gpu_topk_sweep_total_latency.png
figures/phase2_gpu_topk_sweep_throughput.png
figures/phase2_gpu_topk_sweep_major_faults.png
figures/phase2_gpu_topk_sweep_minor_faults.png
figures/phase2_gpu_topk_sweep_swap_free_delta.png
figures/phase2_gpu_topk_sweep_swap_cached_delta.png
figures/phase2_gpu_topk_sweep_total_latency_by_task.png
figures/phase2_gpu_topk_sweep_stage_latency.png
figures/phase2_gpu_topk_sweep_major_fault_breakdown.png
figures/phase2_gpu_topk_sweep_prefetch_cost.png
```

### 初步觀察

- 降低 top_k 確實能明顯降低 prefetch 量：top_k=2 約 122MB、top_k=4 約
  243MB、top_k=8 約 487MB。
- top_k=4 的 mean major faults 最低，比 no-RPP 少約 39.8%，但 total latency
  仍比 no-RPP 慢約 20.8%。
- top_k=8 是三個 RPP 設定裡唯一讓 continuation latency 和 continuation
  major faults 都下降的設定，但 prefetch 量最大，overall 仍慢約 16.8%。
- top_k=2 / top_k=4 雖然降低整體 major faults，卻沒有降低 continuation
  latency；這表示只看 request-level major faults 不夠，還要看是否真的命中
  continuation 需要的 expert pages。
- 這組 sweep 支持目前判斷：問題不只是 `recall@8`，而是 RPP Python-side
  overhead、prefetch timing、prefetch precision/placement 與 page-cache
  behavior 的共同結果。
