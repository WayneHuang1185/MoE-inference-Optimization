# RPP-GPU：GPU Expert Paging / Cache 實驗

這個資料夾用來做新的 RPP-GPU 架構實驗。目標不是延續前一版「只把 expert page 讀進 CPU page cache」的 RPP，而是往下面這個方向前進：

```text
RPP 預測 upcoming experts
-> CPU 背景讀 SSD / page cache
-> staging 到 host memory
-> async 搬到 GPU VRAM expert cache
-> MoE expert matmul 儘量在 GPU 上算
```

目前先改用 `--cpu-moe` 作為基底。原因是 8GB VRAM 放不下完整 MoE，但 `--cpu-moe` 會讓非 MoE 的 dense/attention 權重盡量留在 GPU，同時讓 MoE expert 權重保留在 CPU mapped memory，之後才有空間做我們自己的 VRAM expert cache。

## 實驗環境

- 模型：`/home/hazcashi/lab/model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- 推論程式：`/home/hazcashi/lab/llama.cpp/build/bin/llama-server`
- GPU：本機 NVIDIA GeForce RTX 4060 Laptop GPU，VRAM 8GB
- 預設 prompt：`/home/hazcashi/lab/experience/RPP/analyze/prompts.jsonl`
- 預設參數：
  - `--ngl 999`
  - `--cpu-moe`
  - `--mmap`
  - `--no-warmup`
  - `GGML_OP_OFFLOAD_MIN_BATCH=1`
  - `ctx = 2048`
  - `n_predict = 32`
  - `temperature = 0.0`

## 目前架構假設

llama.cpp 裡面有一段重要機制：當 MoE expert weight 留在 CPU host memory，但某個 `MUL_MAT_ID` 被 scheduler offload 到 GPU 時，它不會把整個 expert tensor 搬到 GPU，而是會根據該次 router/top-k 實際選到的 expert id，只搬 selected experts 的連續 ranges。

因此 Phase 1 先不做自己的 cache，而是先觀測現有 llama.cpp 行為：

```text
router/top-k selected experts
-> llama.cpp scheduler 找出 used experts
-> 只把 used expert ranges 從 host 搬到 GPU backend
-> GPU 執行 MoE mul_mat_id
```

這件事很關鍵，因為它代表 `--cpu-moe` 不是單純「MoE 全部在 CPU 算」。實際是否 offload、搬了多少、每層搬哪些 expert，都需要先量出來。

目前 smoke test 的實際觀察分成兩種：

```text
--ngl 999 --cpu-moe
GGML_OP_OFFLOAD_MIN_BATCH=32，也就是 llama.cpp 預設值
router / top-k 等小 MoE 節點：CUDA0
大型 MoE expert MUL_MAT_ID：CPU
selected-expert H2D copy event：0
```

```text
--ngl 999 --cpu-moe
GGML_OP_OFFLOAD_MIN_BATCH=1
router / top-k 等小 MoE 節點：CUDA0
大型 MoE expert MUL_MAT_ID：幾乎全部 CUDA0
selected-expert H2D copy event：會觸發
```

第二種模式已經能做到「expert matmul 在 GPU 上執行」，但目前是每次用到 expert 就現場從 CPU host memory 搬到 GPU，因此 H2D payload 很大，smoke test 觀察到單一 request 約 25GB selected-expert payload。這會比純 CPU-MoE 更慢，並不是最終目標。真正要做的是在這條已驗證的 offload path 上加 RPP prefetch / GPU expert cache，避免每次重複搬。

另外，這裡先不強制把原生 router 放到 CPU。原因是 router/gate/top-k 目前在 CUDA0 已經很小，強制 CPU 會多產生 activation/weight 往返，通常不會是有效瓶頸。RPP predictor 可以獨立在 CPU 跑；原生 router 則先保留在 CUDA，確保真正輸出的 expert selection 仍由模型本身決定。

目前保留的 smoke 結果：

```text
results/phase1_cpu_moe_trace_smoke_20260623_143525.summary.md
  - GGML_OP_OFFLOAD_MIN_BATCH=32
  - MoE MUL_MAT_ID：CPU 4080
  - H2D copy events：0

results/phase1_expert_gpu_offload_smoke_20260623_145343.summary.md
  - GGML_OP_OFFLOAD_MIN_BATCH=1
  - MoE MUL_MAT_ID：CUDA0 4077 / CPU 3
  - H2D copy events：4077
  - H2D payload：約 25.6GB
```

## Phase 1：CPU-MoE GPU Offload Trace

Phase 1 已實作的內容：

- 在 llama.cpp `ggml-backend.cpp` 增加可選 trace hook。
- 只有設定 `GGML_MOE_OFFLOAD_TRACE=/path/file.jsonl` 時才會輸出。
- 不改變原本 inference/scheduler/copy 決策。
- trace event 會帶上目前 llama decode / ubatch metadata：
  - `llama_decode_index`
  - `llama_ubatch_index`
  - `llama_ubatch_n_tokens`
  - `llama_ubatch_n_seqs`
  - `llama_ubatch_pos_min` / `llama_ubatch_pos_max`
- 每次 MoE compute node 會記錄：
  - node name
  - op type
  - 實際執行的 split backend，例如 `CPU` 或 `CUDA0`
  - source tensor 名稱、buffer、是否 host buffer
- 每次 selected-expert copy 真的發生時會記錄：
  - source backend / destination backend
  - MoE node name
  - expert tensor name
  - selected expert ids
  - 被合併後的 copy ranges
  - payload bytes
  - 含 padding 的實際 enqueue bytes
  - async copy API enqueue 耗時

Phase 1 尚未做的內容：

- 尚未實作 RPP-driven prefetch 到 VRAM。
- 尚未實作持久化 GPU expert cache。
- 尚未有 cache hit/miss，因為 Phase 1 還沒有自訂 cache。
- 目前的 `enqueue_us_total` 只是呼叫 backend async copy API 的排隊時間，不是 GPU H2D copy 完成時間。若要量精準 H2D latency，下一步需要加 CUDA event 或強制同步量測，但那會影響效能。
- 如果 `H2D trace events = 0`，代表 selected-expert copy 路徑沒有觸發，不代表 trace 壞掉；請同時看 `MoE MMID nodes` 與 `compute backends`。

## 使用方式

先重新 build llama.cpp，讓 trace hook 進到 binary：

```bash
cd /home/hazcashi/lab/llama.cpp
cmake --build build -j2
```

跑 1 個 prompt smoke test：

```bash
cd /home/hazcashi/lab/experience/RPP-GPU
/home/hazcashi/lab/.venv/bin/python run_phase1_cpu_moe_trace.py \
  --config local_config.json \
  --max-prompts 1 \
  --repeat 1 \
  --wait-timeout-s 300
```

這會使用 `local_config.json` 裡的 `op_offload_min_batch = 1`，也就是 expert-GPU offload smoke。若要回到 llama.cpp 預設 offload threshold，可加：

```bash
--op-offload-min-batch 32
```

預設 trace detail 是 `large`，只記錄大型 expert matmul 與 router/top-k 關鍵節點。如果需要完整 `ffn_moe_*` 節點，可以加：

```bash
--trace-detail all
```

跑正式 20 prompts：

```bash
cd /home/hazcashi/lab/experience/RPP-GPU
/home/hazcashi/lab/.venv/bin/python run_phase1_cpu_moe_trace.py \
  --config local_config.json \
  --max-prompts 20 \
  --repeat 3 \
  --wait-timeout-s 300
```

輸出位置：

```text
results/phase1_cpu_moe_trace_*.jsonl
results/phase1_cpu_moe_trace_*.summary.csv
results/phase1_cpu_moe_trace_*.summary.md
results/traces/*.trace.jsonl
results/logs/*.server.log
```

## 結果欄位

request JSONL 裡每筆 request 會包含：

- `total_s`：整體 completion latency
- `ttft_s`：time to first token
- `tokens_per_s`：stream chunk throughput
- `major_faults` / `minor_faults`
- `read_bytes_delta`：Linux `/proc/<pid>/io` 的 disk read bytes 變化
- `moe_compute_events`：被 trace 到的 MoE compute node 數量
- `moe_compute_backends`：MoE trace node 實際出現過的 backend
- `moe_mul_mat_id_events`：大型 expert `MUL_MAT_ID` 節點數量
- `moe_mul_mat_id_by_backend`：大型 expert matmul 分別在哪些 backend 執行
- `op_offload_min_batch`：CUDA backend 是否願意 offload host-weight op 的 batch threshold
- `moe_trace_events`：selected-expert copy event 數量
- `moe_trace_payload_bytes`：不含 padding 的 expert payload bytes
- `moe_trace_copied_bytes_with_padding`：含 llama.cpp padding 的 enqueue bytes
- `moe_trace_copied_ranges`：連續 expert range copy 次數
- `moe_trace_used_experts_sum`：各 event 的 unique used experts 加總
- `moe_trace_by_tensor`：依 `up/gate/down/gate_up` 統計
- `moe_trace_top_layers`：copy bytes 最大的 layers

## Phase 2：Offline Oracle RPP Hints

Phase 2 先做離線 oracle，不直接改 runtime prefetch。做法是把 llama.cpp router 真正選到的 expert 當成 RPP 預測結果，因此這份 hint 的 prediction accuracy 等於 100%。

目的：

```text
先回答「如果 RPP 完全預測正確，RPP + GPU expert cache 理論上有沒有用」
再決定值不值得往 runtime prefetch / overlap 實作
```

這個版本的 prediction 單位是 ubatch，不是整個 request。新 trace 會直接記錄 `llama_decode_index` 與 `llama_ubatch_index`；離線工具會依照 trace 的實際出現順序產生 hint。如果拿舊 trace 來跑，因為舊 trace 沒有真實 ubatch 欄位，工具會用 layer 從高層回到低層的位置推斷下一個 ubatch，並在 summary 裡標成 `inferred`。

產生 offline oracle hints：

```bash
cd /home/hazcashi/lab/experience/RPP-GPU
/home/hazcashi/lab/.venv/bin/python build_oracle_hints.py \
  --result results/phase1_expert_gpu_offload_smoke_20260623_145343.jsonl \
  --output-prefix offline_oracle_rpp \
  --cache-mb 0,256,512,1024,2048,4096,6144
```

如果是新跑出來的正式 Phase 1 結果，把 `--result` 換成新的 result JSONL：

```bash
/home/hazcashi/lab/.venv/bin/python build_oracle_hints.py \
  --result results/<正式 Phase 1 result>.jsonl \
  --output-prefix offline_oracle_rpp_formal \
  --cache-mb 0,256,512,1024,2048,4096,6144
```

如果要合併多個 result，就重複寫多次 `--result`。

輸出位置：

```text
results/oracle_hints/*.hints.jsonl
results/oracle_hints/*.cache.csv
results/oracle_hints/*.summary.md
```

hint JSONL 每筆 `oracle_hint` 代表一次 selected-expert copy event，內容包含：

- request / prompt / repeat metadata
- `llama_decode_index` / `llama_ubatch_index`
- layer 與 tensor kind，例如 `gate/up/down`
- 真實 selected expert ids
- expert size、payload bytes、copy ranges

cache simulation 的解讀：

- `cache_mb = 0`：沒有持久化 VRAM expert cache，每次用到都視為 miss。
- `miss MB`：perfect predictor 仍然需要搬進 GPU cache 的 expert payload。
- `saved MB`：相對於現在 on-demand selected-expert copy，理論上可因 cache hit 少搬的 payload。
- 如果 cache 很小且 hit rate 很低，RPP 的主要價值只剩把 H2D 搬運提前做 overlap。
- 如果 cache 有明顯 hit，RPP + VRAM expert cache 才可能同時減少 H2D bytes 和等待時間。

## Phase 3A：Offline Real-RPP First-Token 分析

Phase 3A 將 Phase 2 的 100% oracle 換成真實 RPP checkpoint，但仍然是離線分析，還沒有把 RPP/cache 接進 runtime。目的不是宣稱 latency 已改善，而是先回答：

```text
真實 RPP predictor 用 prompt + first generated token 預測 continuation experts 時，
相對 demand-only GPU expert cache 能不能額外降低 H2D miss payload？
top-k 8 會不會因為 payload 太大而來不及傳？
```

本階段刻意不處理 prefill。流程如下：

```text
1. 讀取已跑完的 RPP-GPU formal trace result。
2. 從每筆 request 的 generated preview 取回 first generated token id。
3. 使用 prompt + first token 跑真實 RPP d64 checkpoint。
4. 產生 top-k 2 / 4 / 8 的 predicted expert set。
5. 只保留 trace 中 pos >= prompt_token_count 的 continuation demands。
6. 模擬不同 VRAM expert cache capacity。
7. 額外估計 RPP forward + H2D prefetch 是否能在 demand time 前完成。
```

這裡的 `top-k 0` 代表沒有 RPP prefetch、只有 demand-loaded persistent GPU cache 的 baseline。這個 baseline 很重要，因為只要有 GPU expert cache，重複使用的 experts 本來就會產生 cache hit；real-RPP 的價值要看它能不能在這個 demand-only cache 之上再降低 miss payload。

執行方式：

```bash
cd /home/hazcashi/lab/experience/RPP-GPU
/home/hazcashi/lab/.venv/bin/python build_real_rpp_offline.py \
  --result results/phase2_oracle_ubatch_trace_formal_20260624_171414.jsonl \
  --top-k 2,4,8 \
  --cache-mb 0,1024,2048,4096,6144 \
  --h2d-gbps 12 \
  --output-prefix real_rpp_first_token_offline_formal
```

正式結果：

```text
results/real_rpp_offline/real_rpp_first_token_offline_formal_0625_2251.summary.md
results/real_rpp_offline/real_rpp_first_token_offline_formal_0625_2251.summary.csv
results/real_rpp_offline/real_rpp_first_token_offline_formal_0625_2251.predictions.jsonl
results/real_rpp_offline/real_rpp_first_token_offline_formal_0625_2251.requests.csv
```

主要圖表：

```text
results/figures/real_rpp_h2d_miss_by_topk_cache.png
results/figures/real_rpp_cache_hit_rate_by_topk_cache.png
results/figures/real_rpp_late_payload_4gb.png
results/figures/real_rpp_prediction_recall_4gb.png
results/figures/real_rpp_false_positive_payload_4gb.png
```

正式 60 requests 的 continuation-only 結果摘要如下：

| top-k | cache | hit rate | H2D miss GB | reduction | pred recall(payload) | false positive GB |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4GB | 68.9% | 342.4 | 68.9% | 0.0% | 0.0 |
| 2 | 4GB | 69.4% | 336.5 | 69.4% | 4.1% | 2.8 |
| 4 | 4GB | 70.0% | 330.4 | 70.0% | 7.5% | 5.3 |
| 8 | 4GB | 71.0% | 319.2 | 71.0% | 13.2% | 11.1 |

解讀：

- continuation-only baseline demand payload 為 1100.7GB，低於 Phase 2 oracle 的 total 1572.8GB，因為 Phase 3A 排除了 prefill。
- demand-only 4GB cache 已可將 miss payload 降到 342.4GB。
- real RPP top-k 8 + 4GB cache 進一步降到 319.2GB，額外節省約 23.2GB，代表真實 RPP 有幫助，但幅度遠小於 oracle upper bound。
- top-k 8 的 prediction recall by payload 約 13.2%，false positive payload 約 11.1GB。這表示 first-token RPP 對「整段 continuation」的覆蓋率仍有限。
- 在假設 H2D bandwidth = 12GB/s 下，top-k 8 的平均 prefetch finish time 約 56.9ms，late predicted payload 只有約 0.4GB；目前更大的問題不是傳不完，而是 first-token RPP 預測到的 true continuation demand 不夠多。

## Phase 3B：Runtime Demand-only GPU Expert Cache

Phase 3B 已經把第一版真實 runtime GPU expert cache 接進 selected-expert copy path。這一版先不接 RPP，也不做 async prefetch；目的先建立一個可以實測的 demand-only cache baseline，確認「持久化 VRAM expert cache」本身是否真的能減少 host-to-device payload。

目前 runtime cache 的流程如下：

```text
1. router/top-k 產生本次真正 selected experts。
2. scheduler 逐一檢查 selected expert 是否已在 GPU expert cache。
3. cache hit:
   GPU cache slot -> input_cpy 做 device-to-device staging copy。
4. cache miss:
   CPU mmap expert weight -> GPU cache slot 做 H2D copy。
   GPU cache slot -> input_cpy 做 device-to-device staging copy。
5. GPU 使用原本的 input_cpy 執行 MUL_MAT_ID。
```

這一版是真實 runtime cache，不是離線模擬；hit 時不會再從 CPU host memory 搬該 expert payload。但它仍然不是最終 direct-read cache，因為目前 compute graph 還是讀原本的 `input_cpy`，所以 cache hit 仍需要一段 GPU-to-GPU D2D staging copy。最終若要把這段也拿掉，需要讓 MoE matmul 直接讀持久化 cache slot，這會是更侵入式的 graph/backend 改動。

實作細節：

- cache key：`(host weight data pointer, expert id)`，避免 decode graph 重建時 tensor metadata pointer 改變造成 false miss。
- cache policy：per CUDA backend 的 LRU cache。
- cache slot：固定 slot size，遇到更大的 expert tensor 時會同步 backend 並重建成較大的 slot。
- 啟用方式：
  - 直接設環境變數：`GGML_MOE_EXPERT_CACHE_MB=2048`
  - 或用 runner 參數：`--moe-expert-cache-mb 2048`
- 關閉方式：不設定 env，或 runner 加 `--moe-expert-cache-mb 0`。
- trace 新增欄位：
  - `demand_payload_bytes`
  - `expert_cache_hits`
  - `expert_cache_misses`
  - `expert_cache_bypasses`
  - `expert_cache_evictions`
  - `expert_cache_h2d_payload_bytes`
  - `expert_cache_d2d_bytes`
  - `expert_cache_status`

執行 smoke：

```bash
cd /home/hazcashi/lab/experience/RPP-GPU

# 關閉 runtime cache，確認 baseline path
/home/hazcashi/lab/.venv/bin/python run_phase1_cpu_moe_trace.py \
  --config local_config.json \
  --max-prompts 1 \
  --repeat 1 \
  --moe-expert-cache-mb 0 \
  --output-prefix runtime_expert_cache_cli_smoke_off

# 啟用 2GB runtime cache
/home/hazcashi/lab/.venv/bin/python run_phase1_cpu_moe_trace.py \
  --config local_config.json \
  --max-prompts 1 \
  --repeat 1 \
  --moe-expert-cache-mb 2048 \
  --output-prefix runtime_expert_cache_cli_smoke_2gb
```

目前 1 prompt smoke 結果：

| cache | total s | TTFT s | tok/s | demand MB | actual H2D MB | cache hit | D2D MB | major faults | read MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0MB | 6.663 | 2.992 | 4.802 | 25581.7 | 25581.7 | 0.0% | 0.0 | 7571 | 6093.9 |
| 2048MB | 6.079 | 2.935 | 5.264 | 25581.7 | 14407.7 | 43.7% | 25602.3 | 8265 | 6007.4 |

解讀：

- 2GB runtime cache 確實讓 actual H2D payload 從 25.58GB 降到 14.41GB，約少 43.7%。
- 這裡的 `D2D MB` 不是 CPU/GPU 傳輸，而是 cache hit/miss 後都要 staging 回 `input_cpy` 的 device-to-device copy。
- 512MB cache 在 smoke 中幾乎沒有 hit，原因是 slot 放大後只剩約數百個 expert slots，還沒等到 reuse 就被 LRU 淘汰；這和離線分析中「小 cache 幾乎沒有明顯收益」一致。
- 目前 latency 只小幅改善，因為這版仍有 D2D staging，且沒有把 H2D 與 GPU compute overlap。

## Phase 4：Runtime Demand-only Cache 正式實驗

Phase 4 已完成 20 prompts、repeat 3 的正式 runtime cache sweep。這裡測的是「不含 RPP prefetch」的 demand-only GPU expert cache，因此可以作為後續 RPP async prefetch 的 runtime baseline。

執行方式：

```bash
cd /home/hazcashi/lab/experience/RPP-GPU

# 0MB: 關閉 runtime cache
/home/hazcashi/lab/.venv/bin/python run_phase1_cpu_moe_trace.py \
  --config local_config.json \
  --max-prompts 20 \
  --repeat 3 \
  --wait-timeout-s 300 \
  --request-timeout-s 900 \
  --moe-expert-cache-mb 0 \
  --output-prefix phase4_runtime_cache_0mb_formal

# 2GB runtime cache
/home/hazcashi/lab/.venv/bin/python run_phase1_cpu_moe_trace.py \
  --config local_config.json \
  --max-prompts 20 \
  --repeat 3 \
  --wait-timeout-s 300 \
  --request-timeout-s 900 \
  --moe-expert-cache-mb 2048 \
  --output-prefix phase4_runtime_cache_2048mb_formal

# 4GB runtime cache
/home/hazcashi/lab/.venv/bin/python run_phase1_cpu_moe_trace.py \
  --config local_config.json \
  --max-prompts 20 \
  --repeat 3 \
  --wait-timeout-s 300 \
  --request-timeout-s 900 \
  --moe-expert-cache-mb 4096 \
  --output-prefix phase4_runtime_cache_4096mb_formal
```

合併分析與畫圖：

```bash
cd /home/hazcashi/lab
/home/hazcashi/lab/.venv/bin/python experience/RPP-GPU/plot_runtime_cache_formal.py \
  --result results/phase4_runtime_cache_0mb_formal_20260625_231637.jsonl \
  --result results/phase4_runtime_cache_2048mb_formal_20260625_233533.jsonl \
  --result results/phase4_runtime_cache_4096mb_formal_20260625_235359.jsonl
```

正式結果：

| cache | requests | total s | TTFT s | tok/s | demand MB/request | actual H2D MB/request | H2D reduction | cache hit | D2D MB/request | major faults | read MB/request |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0MB | 60 | 7.753 | 3.856 | 4.026 | 26213.6 | 26213.6 | 0.0% | 0.0% | 0.0 | 9732 | 7194.5 |
| 2GB | 60 | 7.308 | 3.929 | 4.316 | 26213.6 | 16415.0 | 37.4% | 37.2% | 26234.6 | 10127 | 7062.7 |
| 4GB | 60 | 7.014 | 3.958 | 4.517 | 26213.6 | 12739.4 | 51.4% | 51.2% | 26234.6 | 9942 | 6990.3 |

依 prompt 類型觀察：

| task | cache | total s | tok/s | actual H2D MB/request | H2D reduction | cache hit |
|---|---:|---:|---:|---:|---:|---:|
| Text | 4GB | 5.813 | 5.529 | 10702.7 | 57.9% | 58.0% |
| Code | 4GB | 6.583 | 4.876 | 11616.4 | 55.3% | 55.3% |
| Commonsense | 4GB | 6.519 | 4.928 | 11728.3 | 54.3% | 54.3% |
| Math | 4GB | 8.480 | 3.805 | 15507.5 | 45.3% | 45.2% |
| MC | 4GB | 7.673 | 3.448 | 14142.1 | 44.9% | 43.1% |

輸出檔案：

```text
results/phase4_runtime_cache_formal_comparison.md
results/phase4_runtime_cache_formal_comparison.csv
results/phase4_runtime_cache_formal_by_task.csv
results/figures/phase4_runtime_cache_actual_h2d.png
results/figures/phase4_runtime_cache_hit_rate.png
results/figures/phase4_runtime_cache_latency.png
results/figures/phase4_runtime_cache_throughput.png
results/figures/phase4_runtime_cache_d2d.png
results/figures/phase4_runtime_cache_task_h2d.png
results/figures/phase4_runtime_cache_task_latency.png
results/figures/phase4_runtime_cache_task_hit_rate.png
```

解讀：

- 4GB runtime cache 在 RTX 4060 Laptop 8GB VRAM 上可以完成正式測試，沒有 OOM。
- 2GB cache 將 actual H2D 從 26.21GB/request 降到 16.42GB/request，約少 37.4%。
- 4GB cache 將 actual H2D 降到 12.74GB/request，約少 51.4%。
- latency 改善存在但小於 H2D reduction：0MB 為 7.753s，4GB 為 7.014s。主要原因是這版仍需 D2D staging，且尚未做 RPP-driven async prefetch / compute-copy overlap。
- text/code/commonsense 的 hit rate 高於 math/multiple-choice，代表不同 prompt 類型會造成不同 expert reuse pattern；後續 RPP prefetch 需要分 task type 觀察，不應只看整體平均。

## Phase 5：Runtime RPP Hint Admission Smoke

Phase 5 先做一個比離線模擬更接近 runtime、但仍不是最終 async prefetch 的中間版本。流程如下：

```text
1. 先跑 first-token request。
2. Python 端用 prompt + first token 跑真實 RPP checkpoint。
3. 將 RPP top-k predictions 寫成 hint file：
   layer tensor_kind expert_id
4. continuation request 啟動 llama-server 時設定：
   GGML_MOE_RPP_HINTS=/path/to/hints.txt
5. C++ selected-expert copy path 讀 hint file。
6. 對當前 tensor 的 hinted experts 先 admit 到 GPU expert cache。
7. 再用真實 router selected experts 執行原本 demand path。
```

這個版本的重點是驗證「RPP prediction 能不能真的進入 C++ runtime GPU expert cache」。它還沒有做到 background worker，也沒有把 H2D copy 和 GPU compute overlap；hint admission 發生在對應 tensor 的 selected-expert copy path 裡，所以仍可能增加同步 H2D 成本。

C++ 端新增的環境變數：

```bash
GGML_MOE_RPP_HINTS=/home/hazcashi/lab/experience/RPP-GPU/results/rpp_hints/<file>.hints.txt
```

hint file 格式：

```text
# layer tensor_kind expert_id
12 gate 87
12 up 87
12 down 87
```

執行 smoke：

```bash
cd /home/hazcashi/lab/experience/RPP-GPU

/home/hazcashi/lab/.venv/bin/python run_runtime_rpp_hint.py \
  --config local_config.json \
  --max-prompts 1 \
  --repeat 1 \
  --top-k 8 \
  --moe-expert-cache-mb 4096 \
  --output-prefix phase5_runtime_rpp_hint_smoke
```

top-k sweep smoke 已跑 rank top-k 2 / 4 / 8，以及 FTO-filtered top-8 admission。每個設定都會跑 matched demand-only continuation 與 RPP-hint continuation。

FTO 版本的做法是：先從既有 formal trace 建立每層 expert frequency table，再從 RPP top-8 candidates 中挑歷史上較常被用到的 experts。`admit-1` 代表每層最多 admit 1 個 expert，`admit-2` 代表每層最多 admit 2 個 experts。

| variant | RPP ms | hint slots | demand-only s | RPP hint s | latency delta | demand-only total H2D MB | RPP demand H2D MB | RPP hint H2D MB | RPP total H2D MB | total H2D delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FTO top-8/admit-1 | 1148.2 | 40 | 5.684 | 5.575 | -0.109 | 9880.2 | 9794.7 | 180.7 | 9975.4 | +95.2 |
| FTO top-8/admit-2 | 1271.0 | 79 | 5.745 | 6.065 | +0.321 | 9880.2 | 9741.3 | 358.3 | 10099.6 | +219.3 |
| Rank top-2 | 1103.6 | 80 | 5.852 | 5.377 | -0.474 | 9880.2 | 9862.1 | 366.6 | 10228.6 | +348.4 |
| Rank top-4 | 1232.7 | 160 | 5.321 | 6.225 | +0.904 | 9880.2 | 10013.2 | 740.8 | 10753.9 | +873.7 |
| Rank top-8 | 1102.3 | 320 | 5.417 | 6.030 | +0.613 | 9880.2 | 10146.3 | 1531.8 | 11678.2 | +1798.0 |

輸出檔案：

```text
results/phase5_runtime_rpp_hint_smoke_comparison.md
results/phase5_runtime_rpp_hint_smoke_comparison.csv
results/figures/phase5_runtime_rpp_hint_total_h2d.png
results/figures/phase5_runtime_rpp_hint_latency.png
results/figures/phase5_runtime_rpp_hint_extra_h2d.png
```

解讀：

- runtime RPP hint 入口已驗證成功：trace 中可看到 `rpp_hint_candidates`、`rpp_hint_h2d_payload_bytes` 與 `rpp_hint_hit_rate`。
- naive rank top-k 2 / 4 / 8 都讓 total H2D 增加，原因是 hinted experts 會額外搬進 cache，但對後續 true selected demand 的降低很有限。
- FTO-filtered admission 能明顯壓低額外 H2D：FTO admit-1 只增加約 +95MB，優於 rank top-2 的 +348MB。
- FTO admit-1 單次 latency 也略快，但這仍是 1 prompt smoke，不能當作穩定改善結論。
- RPP forward 在本機 CPU 約 1.1 到 1.2 秒，若未來要進真正 runtime path，需要降低 RPP 開銷或讓它和 GPU compute overlap。
- 下一步可以優先針對 FTO admit-1 做小型 repeat，確認它是否穩定；但真正要產生明顯收益，仍需要 background async prefetch 或更細的 admission policy。

## 後續 Phase

RPP hint 已經可以進入 runtime cache path。下一步會分成兩條：

1. 先對目前最好的 FTO top-8/admit-1 做小型 repeat，例如 5 prompts × 3 repeat，確認 smoke 中的趨勢是否穩定。
2. 再把 admission 從目前的 selected-expert copy path 移到真正的 background async prefetch path，目標是從 demand-only cache 進一步變成 prefetch cache：

```text
CPU RPP 預測 future experts
-> cache manager 檢查 missing predicted experts
-> background H2D prefetch 到 VRAM cache
-> GPU compute 到 MoE layer 時檢查 true selected experts
-> hit 直接 D2D staging / miss 同步補搬
```

再下一步是 direct-read cache，讓 GPU MoE matmul 直接讀持久化 cache slot，避免 hit 時仍需 D2D staging。這部分會比目前 Phase 3B 更大改，因為需要改 compute graph 讀取的 expert weight 來源。

## Phase 6：Qwen Online RPP-GPU Runtime Port

Phase 6 改用 `/home/hazcashi/lab/rpp_runtime_implementation/llama.cpp` 這份 runtime implementation 作為完整 RPP-GPU 實作基底。這份 runtime 和前面 Phase 5 的差別是：它不只是用環境變數把 RPP hint 插進 selected-expert copy path，而是有明確的 online sidecar、page map、GPU expert cache、correction path 與 GPU compute path。

目前已經把它從原本較偏 Gemma-style 的 expert layout 改成能支援 Qwen3.6 MoE 的 separate expert tensors：

```text
blk.N.ffn_gate_exps.weight
blk.N.ffn_up_exps.weight
blk.N.ffn_down_exps.weight
```

新增或修改的重點：

- `scripts/build_expert_page_map.py`：從 Qwen GGUF 直接產生 expert byte range page map。
- `llama-rpp-prefetch.*`：page map parser 支援 `gate`、`up`、`down` separate components。
- `llama-rpp-gpu-cache.*`：GPU expert cache 支援 separate gate/up/down layout。
- `llama-graph.cpp`：MoE compute graph 可在 Qwen separate expert layout 下改讀 GPU cache tensor。
- `llama-rpp-runtime.cpp`：ubatch 結束後才釋放 cache protection，避免同一個 CUDA graph 內 cache slot 被過早覆寫。
- `scripts/rpp_predict_prompt.py`：RPP predictor 改成讀 config 中的 vocab size，避免只適用 Gemma vocab。

build 指令：

```bash
cd /home/hazcashi/lab/rpp_runtime_implementation/llama.cpp
cmake -S . -B build-rpp-cuda124 -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build-rpp-cuda124 --target llama-server test-rpp-args test-rpp-runtime test-rpp-prefetch test-rpp-gpu-cache -j "$(nproc)"
```

產生 Qwen expert page map：

```bash
cd /home/hazcashi/lab/rpp_runtime_implementation
python3 scripts/build_expert_page_map.py \
  --model /home/hazcashi/lab/model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --out outputs/qwen36_rpp_gpu/expert_page_map.csv
```

page map smoke 結果：

| item | value |
|---|---:|
| rows | 30720 |
| layers | 40 |
| experts / layer | 256 |
| components | down / gate / up |
| layer 0 expert 0 total bytes | 1900544 |

可直接重跑 online RPP-GPU smoke：

```bash
cd /home/hazcashi/lab/experience/RPP-GPU
chmod +x run_qwen_online_rpp_gpu_smoke.sh
./run_qwen_online_rpp_gpu_smoke.sh
```

可調參數範例：

```bash
TOP_K=2 CACHE_MIB=1024 N_PREDICT=6 ./run_qwen_online_rpp_gpu_smoke.sh
```

目前 06/26 smoke 使用的參數：

```text
--ngl 999
--cpu-moe
--rpp-mode online
--rpp-sidecar-url http://127.0.0.1:18181
--rpp-prefetch-top-k 2
--rpp-gpu-correction on
--rpp-gpu-compute on
--rpp-gpu-cache-mib 1024
--rpp-gpu-staging-mib 64
--rpp-gpu-copy-workers 1
--rpp-gpu-queue-policy deadline
```

目前 06/26 smoke 結果：

| metric | value |
|---|---:|
| completion status | OK |
| tokens evaluated | 9 |
| tokens predicted | 6 |
| prompt ms | 1747.031 |
| predicted ms | 1238.781 |
| predicted tok/s | 4.843 |
| trace events | 200 |
| GPU correction success | 200 / 200 |
| prediction found | 200 / 200 |
| prefetched experts | 400 |
| hit experts | 183 |
| missing experts | 1417 |
| cache slots | 388 |
| max resident entries | 388 |
| sidecar requests | 6 |
| last CPU RPP inference ms | 3.345 |

輸出檔案：

```text
/home/hazcashi/lab/experience/RPP-GPU/results/phase6_qwen_online_rpp_gpu_smoke_0626.summary.md
/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/expert_page_map.csv
/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/rpp_gpu_online_trace.jsonl
/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/sidecar_metrics.jsonl
```

解讀：

- 這次已經不是 offline simulation；真實 RPP sidecar prediction 有進入 runtime，`prediction_found = 200 / 200`。
- Qwen 的 selected expert GPU compute/correction path 有跑通，`gpu_correction_success = 200 / 200`。
- 這仍是 smoke test，不是正式效能結論；目前 top-k=2 只預抓少量 experts，所以多數 true selected experts 仍要 on-demand 補搬。
- 先前 256MiB cache 版本曾觸發 CUDA illegal memory access；目前已修成 ubatch 結束才釋放 cache protection，並以 1GiB cache 通過 smoke。小 cache 需要額外重跑驗證。

### Phase 6 Formal：20 prompts x 3

正式測試已完成 20 prompts x repeat 3，每個 scenario 共 60 requests。測試參數如下：

```text
n_predict = 32
ctx_size = 512
--ngl 999
--cpu-moe
GPU expert cache = 1024 MiB
GPU staging = 64 MiB
copy workers = 1
queue policy = deadline
```

比較場景：

| scenario | 含義 |
|---|---|
| `rpp-off-native` | 不啟用 RPP runtime，作為 native `--cpu-moe` 參考 |
| `demand-cache-1g` | 啟用 GPU expert cache/correction，但不給 RPP prediction |
| `online-rpp-top2` | 真實 RPP sidecar，top-k=2 |
| `online-rpp-top4` | 真實 RPP sidecar，top-k=4 |
| `online-rpp-top8` | 真實 RPP sidecar，top-k=8 |

正式結果：

| scenario | requests | latency s | tok/s | prediction coverage | RPP hit | ready-hit | total H2D GB/request | sidecar ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rpp-off-native | 60 | 6.084 | 10.757 | 0.0% | 0.0% | 0.0% | 0.000 | 0.000 |
| demand-cache-1g | 60 | 9.734 | 4.942 | 0.0% | 0.0% | 36.7% | 11.626 | 0.000 |
| online-rpp-top2 | 60 | 11.027 | 4.300 | 100.0% | 8.6% | 36.6% | 16.233 | 4.375 |
| online-rpp-top4 | 60 | 10.768 | 4.264 | 100.0% | 16.2% | 17.2% | 24.134 | 4.419 |
| online-rpp-top8 | 60 | 11.558 | 3.915 | 100.0% | 27.4% | 1.2% | 31.787 | 4.805 |

正式結果輸出：

```text
results/phase6_qwen_online_rpp_gpu_formal_0626_1441.summary.md
results/phase6_qwen_online_rpp_gpu_formal_0626_1441.compact.csv
results/phase6_qwen_online_rpp_gpu_formal_0626_1441.by_task.csv
results/figures/phase6_formal_latency.png
results/figures/phase6_formal_throughput.png
results/figures/phase6_formal_total_h2d_per_request.png
results/figures/phase6_formal_hit_rates.png
results/figures/phase6_formal_latency_by_task.png
```

原始 runtime output：

```text
/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/formal/phase6_qwen_online_rpp_gpu_formal_0626_1441.summary.md
/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/formal/phase6_qwen_online_rpp_gpu_formal_0626_1441.requests.csv
```

正式解讀：

- Phase 6 online path 功能上已正式跑通：top2/top4/top8 都有 `prediction coverage = 100%`，代表真實 RPP sidecar prediction 都有進 runtime。
- 但在目前 1GB GPU expert cache 下，online RPP 還不是有效優化；三個 top-k 都比 `demand-cache-1g` 慢。
- top-k 越大，RPP hit rate 越高，但 total H2D payload 也越大。top8 的 RPP hit rate 最高，但 total H2D 達 31.787GB/request，throughput 最低。
- sidecar CPU inference 平均只有約 4 到 5ms，因此目前主要瓶頸不是 RPP model forward，而是 prefetch admission、cache eviction、H2D 搬運與 ready-hit 不足。
- 下一步不應繼續單純加大 top-k；更值得做的是 FTO/admission policy、larger cache sweep，以及真正 async overlap / direct-read cache。
