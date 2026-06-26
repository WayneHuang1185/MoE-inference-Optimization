# Qwen3.6 MoE / RPP 實驗總報告

最後更新：2026-06-26。本文已包含 CPU/GPU baseline、RPP first-token prefetch、RPP top-k sweep、RPP-GPU selected-expert offload trace、ubatch-aware offline oracle、offline real-RPP first-token 分析、runtime demand-only GPU expert cache formal sweep、runtime RPP hint-admission smoke、Qwen online RPP-GPU formal sweep，以及 prompt task type 分析。

## 摘要

本報告整理 Qwen3.6-35B-A3B MoE 模型在本機筆電環境下的推論瓶頸與 RPP-based IO optimization 實驗。核心情境是：模型 GGUF 約 21GB，系統 RAM 約 14GiB，GPU VRAM 約 8GiB，因此完整模型無法同時常駐 CPU memory 或 GPU VRAM。推論過程必須頻繁在 SSD、Linux page cache、CPU DRAM 與 GPU VRAM 之間搬移 expert weights。

本研究分成兩條主線。第一條是 CPU-side RPP prefetch：用 Routing Path Predictor 預測後續可能使用的 MoE experts，將 expert id 轉成 GGUF byte ranges，提前觸發 page cache 載入。第二條是 RPP-GPU：把 dense/attention 計算放在 GPU，MoE expert weights 保留在 CPU mapped memory，再透過 selected-expert H2D copy、GPU expert cache 與 RPP/FTO hints 嘗試降低重複搬移。

目前結果顯示：

1. CPU-only 推論嚴重受 page fault / IO 限制，平均 latency 約 32s，吞吐量不到 1 tok/s。
2. GPU partial offload 是最直接有效的 baseline，其中 `baseline-gpu-ncpu-moe-32` 平均 total latency 約 1.892s，tok/s 約 17.464。
3. RPP first-token CPU page-cache prefetch 能降低部分 major faults，但同步 prefetch 成本尚未轉成 end-to-end latency improvement。
4. Offline oracle 顯示，如果能有 100% accurate prediction 且搭配 4GB 到 6GB GPU expert cache，H2D payload 有明顯下降空間。
5. Runtime demand-only GPU expert cache 是目前唯一在 component level 顯示明確收益的優化；歷史 Phase 4 sweep 中 4GB cache 可將 actual H2D payload 從 26.21GB/request 降到 12.74GB/request。
6. Runtime RPP hint path 與 online RPP sidecar path 都已接通，但 real-RPP + naive top-k admission / prefetch 目前沒有勝過 baseline。最新 Phase 6 formal rerun 顯示 fresh native baseline 平均 6.023s/request，而 online RPP top2/top4/top8 分別為 11.027s、10.768s、11.558s。

因此目前結論不是「RPP 已經帶來完整加速」，而是更精確地說：RPP-GPU 的基礎路徑、trace、cache、hint admission 與 online sidecar 都已建立完成；目前沒有任何 formal online RPP-GPU 設定勝過最新 baseline。真正有價值的下一步是把 RPP/FTO 從 naive top-k prefetch 推進到 batch-aware admission、async prefetch 與更精準的 frequency/confidence policy。

## 一、研究背景與實驗目標

### 1.1 背景

MoE 模型的推論瓶頸同時包含 compute capability 與 memory capacity。對 dense LLM 來說，每一層權重在每個 token 都會被使用；但對 MoE LLM 來說，每層通常有大量 experts，每個 token 只會經由 router 選到少數 experts。這讓 MoE 模型的參數量可以大幅增加，但也讓權重存取呈現稀疏且資料相依的行為。

在本機推論環境中，這個問題會被進一步放大。Qwen3.6-35B-A3B 的 GGUF 檔案約 21GB，大於可用 RAM，也大於 8GiB VRAM。若使用 mmap 載入模型，OS 只會在 page 被實際訪問時才從 SSD 載入；若使用 GPU offload，VRAM 又不足以容納完整 MoE expert weights。因此每次 router 決定 selected experts 後，系統可能才開始載入或搬移對應權重，導致 compute path 等待 IO 或 H2D transfer。

本研究的核心想法是：如果可以提早知道未來會用到哪些 experts，就有機會把 expert weights 的載入或搬移從 critical path 移出去。RPP 負責預測 routing path；FTO 負責根據歷史 frequency 與預測結果決定哪些 experts 值得進 cache；GPU expert cache 則負責讓已搬入 VRAM 的 experts 在後續 ubatch/request 中重複使用。

核心問題如下：

```text
在 expert weights 無法完全常駐於 CPU memory 或 GPU VRAM 的情況下，
是否能透過預測未來將被 activate 的 experts，
提前載入或搬移對應 weights，
並讓資料搬移與當前模型計算重疊，
進而降低 IO / H2D transfer 對推論 latency 的影響？
```

### 1.2 Research Questions

本報告用以下研究問題組織方法與實驗：

| RQ | 問題 | 對應實驗 |
|---|---|---|
| RQ1 | 在本機 14GiB RAM / 8GiB VRAM 下，Qwen3.6 MoE 推論主要受 compute 還是 IO 限制？ | CPU baseline、GPU baseline、page fault / swap / disk read 觀測 |
| RQ2 | 哪一種 GPU partial offload 設定最適合作為後續 RPP-GPU baseline？ | `--ngl`、`--cpu-moe`、`--n-cpu-moe` sweep |
| RQ3 | RPP 預測出的 experts 是否能降低 CPU page-cache 層級的 major faults？ | first-token RPP prefetch、two-step no-RPP 對照、top-k sweep |
| RQ4 | 若 MoE expert matmul 交給 GPU，selected expert weights 的 H2D transfer 量有多大？ | RPP-GPU selected-expert trace、offload threshold sweep |
| RQ5 | GPU expert cache 是否能在 runtime 中實際降低重複 H2D 搬移？ | Phase 4 demand-only runtime cache formal sweep |
| RQ6 | real-RPP hints 加上 FTO admission 是否能進一步優於 demand-only cache？ | Phase 5 runtime RPP hint admission smoke |
| RQ7 | online RPP sidecar 接進 runtime 後，能否在 formal run 中勝過 baseline？ | Phase 6 Qwen online RPP-GPU formal sweep |
| RQ8 | 下一步應該繼續單 request prefetch，還是轉向 batch-aware / ubatch-aware policy？ | offline ubatch trace、runtime formal、後續 batch-aware 設計 |

### 1.3 本報告貢獻

本報告目前完成的工作可整理為五點：

1. 建立本機 MoE 推論 baseline，量化 CPU-only、GPU partial offload、CPU-MoE 與 partial MoE offload 的差異。
2. 訓練並評估 RPP predictor，確認 real-RPP routing prediction 可用但仍有 false positive / limited recall 的問題。
3. 將 RPP prediction 轉成 GGUF byte-range prefetch，觀察 CPU page-cache 層級的 page fault 與 latency 變化。
4. 建立 RPP-GPU trace、offline oracle、offline real-RPP 與 runtime GPU expert cache 實驗，用來分離「理論上限」、「cache 本身效果」與「RPP 額外效果」。
5. 實作 runtime RPP hint admission 與 FTO policy 的初版，確認 RPP hints 能進入 C++ cache path，但也指出 naive top-k 會增加 H2D，需要更細的 batch-aware policy。

## 二、實驗環境

主要實驗環境如下：

| 項目 | 內容 |
|---|---|
| 作業系統 | Ubuntu Linux，x86_64 |
| CPU | AMD Ryzen 7 7840HS，llama.cpp 偵測約 16 logical CPUs |
| RAM / Swap | RAM 約 14GiB，swap 約 4GiB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 約 8GiB |
| CUDA / Driver | NVIDIA driver 595.71.05，CUDA runtime 13.2 |
| 模型 | `model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| 模型大小 | 約 21GB |
| 推論程式 | `llama.cpp/build/bin/llama-server` |
| llama.cpp version | build 8994，commit `aab68217b` |

由於模型大小大於 RAM，CPU-only mmap 推論容易受到 Linux page cache、major page faults、swap 與 SSD IO 影響。因此所有主要實驗除了 TTFT、total latency、tok/s，也記錄 major/minor faults 與 swap/cache 變化。

## 三、方法

### 3.0 方法設計總覽

本研究不是單純測一個 prefetch API，而是把 MoE 推論拆成「預測」、「資料定位」、「資料搬移」、「計算」、「cache replacement」五個部分來觀察。這樣做的原因是：RPP 的 routing accuracy 只代表模型層面的可能性，最後是否加速仍取決於系統層面是否少搬資料、是否少等 IO、以及搬移是否能和計算重疊。

整體系統可抽象成以下資料路徑：

```text
Prompt / context
-> RPP predictor predicts future expert sets
-> expert mapper converts (layer, expert id) to weight ranges
-> admission policy decides which experts are worth loading
-> cache / prefetch layer moves selected weights
-> main model runs true router and selected expert matmul
-> trace layer records latency, page faults, H2D bytes, cache hit/miss
```

在 CPU-side 實驗中，cache / prefetch layer 指的是 Linux page cache；RPP 預測出的 expert ranges 會被提前讀入 CPU memory，希望降低後續 mmap access 的 major page faults。在 RPP-GPU 實驗中，cache / prefetch layer 指的是 GPU expert cache；RPP 或 oracle hints 會被轉成 GPU cache entries，希望降低 selected expert weights 的 H2D transfer。

本研究中特別區分三種方法，避免把不同層級的收益混在一起：

| 方法 | 預測來源 | 是否真的改變 runtime | 主要目的 |
|---|---|---|---|
| no-RPP baseline | 無 | 是，正常推論 | 建立 latency / faults / H2D 對照 |
| offline oracle / offline real-RPP | trace 或 RPP 離線輸出 | 否，只做 post-analysis simulation | 估計上限、檢查 real-RPP 是否值得接 runtime |
| runtime cache / runtime RPP hint | demand events 或 RPP hint file | 是，進入 C++ cache path | 測試真實 H2D、cache hit 與 latency 變化 |

因此當報告中提到 offline 結果時，它代表「如果 runtime 有這種 cache/prefetch policy，理論上可能省多少資料搬移」；當報告中提到 runtime 結果時，它才代表實際推論路徑被修改並完成端到端量測。

### 3.0.1 術語與量測單位

| 術語 | 定義 |
|---|---|
| expert | MoE layer 中可被 router 選擇的 feed-forward 子網路權重 |
| selected experts | 主模型 router 在某個 token / ubatch / layer 真正選到的 experts |
| RPP | Routing Path Predictor，用 prompt/context 預測每層可能會用到的 experts |
| FTO | Frequency Token Organizer，根據 expert access frequency 與 RPP candidates 決定 admission priority |
| page-cache prefetch | 將 GGUF byte ranges 提前讀進 Linux page cache，目標是降低 major faults |
| H2D | host-to-device transfer，將 CPU memory 中的 expert weights 搬到 GPU VRAM |
| D2D | device-to-device transfer，在 GPU memory 內部從 cache slot 搬到 compute staging buffer |
| demand-only cache | 不使用 RPP，只在 true router 實際需要某 expert 時才載入並保留於 GPU cache |
| RPP hint admission | RPP 預測出 future experts 後，在 true demand 之外額外把 candidates admission 到 GPU cache |
| oracle | 使用主模型真實 selected experts 當作 100% 準確預測，作為理論上限 |
| ubatch | llama.cpp 將 logical batch 切成的實際 compute unit；prefetch 順序應對齊 ubatch execution order |

### 3.1 整體流程

本實驗的方法核心是：先用 RPP 預測模型接下來可能會用到的 MoE experts，再把這些 expert 對應到 GGUF 檔案中的 byte ranges，最後透過 page fault、disk read、latency 與 GPU trace 觀察「預測出來的 expert 是否真的能降低 IO 成本」。

整體流程如下：

```text
prompt
-> tokenizer
-> RPP predictor 預測各 layer 可能用到的 experts
-> 將 (layer, expert id) 轉成 GGUF tensor / byte range
-> 對這些 byte ranges 做 prefetch 或觀察 page fault
-> llama.cpp 正常推論
-> 記錄 TTFT、total latency、major faults、minor faults、disk read、swap/cache
-> 比較 baseline / no-RPP / RPP prefetch / GPU offload
```

這裡的重點不是只看 RPP routing accuracy，而是要驗證 RPP 的預測能不能轉換成實際系統層面的收益，例如 major page faults 下降、TTFT 下降或 GPU H2D 搬移變少。

### 3.2 RPP 資料收集與訓練

RPP 目標是用輕量模型預測 Qwen3.6 每層 MoE 會使用哪些 expert，讓系統可以提前載入或搬移對應權重。Qwen3.6 設定為 40 層 MoE、每層 256 experts、Top-K 8。

資料建立流程如下：

1. 建立 prompt database：來源包含 WikiText、MMLU、GSM8K、HumanEval、MBPP、HellaSwag 等任務，涵蓋文字接續、選擇題、數學推理、程式生成與常識推理。
2. 使用 Qwen3.6 生成 completions：讓模型實際跑過這些 prompts，取得真實推論路徑。
3. dump routing labels：在推論過程中收集每個 token、每一層 MoE router 選到的 expert ids 與 router logits。
4. 打包成 NPZ：每筆資料包含 token sequence、router logits、layer mask 等資訊，供 RPP 訓練使用。
5. 訓練 RPP predictor：輸入 token/prompt 表徵，輸出每層可能使用的 expert 分布，訓練目標包含 BCE 與 KL distillation。

正式資料集包含 10,000 個 NPZ 檔，約 179 萬 tokens；任務分布包含 text continuation、multiple choice、math reasoning、code generation、commonsense。訓練後用 token_recall@8、token_recall@16、batch_level_accuracy@8 等指標評估 RPP 是否能抓到主模型實際會使用的 experts。

### 3.3 Expert Range 對應與 Page Fault 觀測

RPP 預測結果本身只是 `(layer, expert id)` 集合，不能直接改善推論。因此實驗中需要先把 expert id 對應回 GGUF 檔案中的實際權重位置。

處理方式如下：

1. 解析 GGUF tensor metadata，找出每層 MoE expert 對應的 tensor。
2. 建立 expert 到 byte range 的 mapping，例如某一層 expert 的 `up`、`gate`、`down` 或 `gate_up` 權重在 GGUF 檔案中的 offset 與 size。
3. 將 RPP 預測出的多個 experts 合併成待讀取的 byte ranges。
4. 在推論前對這些 ranges 做 prefetch，嘗試讓 Linux page cache 先載入相關 expert weights。
5. 推論時觀察 major faults / minor faults / disk read 是否下降。

Page fault 在這裡被當作 expert IO 成本的 proxy。因為模型使用 mmap 載入 GGUF，如果某個 expert weight 尚未在記憶體或 page cache 中，實際訪問時會造成 major page fault，並觸發磁碟讀取。因此若 RPP prefetch 有效，理論上後續推論訪問 expert weight 時 major faults 應下降。

### 3.4 Baseline 與 RPP Prefetch 比較

推論實驗統一使用固定 prompt set：20 個 prompts，每組 scenario repeat 3 次，共 60 requests。主要 generation 參數為 `n_predict=32`、`temperature=0.0`、`ctx=2048`、`threads=8`。

Baseline 設計分成幾層：

1. CPU cold/warm baseline：確認不使用 GPU、不使用 RPP 時，page cache 是否能自然改善推論。
2. GPU baseline：測試 `--ngl`、`--cpu-moe`、`--n-cpu-moe` 在 8GiB VRAM 下的可行設定。
3. two-step no-RPP：把推論流程拆成 first token 與 continuation，但不做 RPP prefetch，用來排除「兩段式流程」本身造成的影響。
4. RPP first-token prefetch：先用 prompt 跑 RPP，預測 upcoming experts，對其 GGUF ranges 做 prefetch，再進行推論。
5. Top-K sweep：比較 RPP 預測 top-k 2 / 4 / 8 時，prefetch MB、major faults 與 latency 的取捨。

測試情境包含：

| 類別 | scenario |
|---|---|
| CPU baseline | `baseline-cpu-cold`、`baseline-cpu-warm` |
| GPU baseline | `baseline-gpu-ngl10`、`baseline-gpu-cpu-moe` |
| MoE partial offload | `baseline-gpu-ncpu-moe-32` |
| RPP prefetch | `rpp-first-token-cpu`、`rpp-first-token-gpu` |
| no-RPP 對照 | `two-step-no-rpp-cpu`、`two-step-no-rpp-gpu` |
| Top-K sweep | RPP GPU top-k 2 / 4 / 8 |
| RPP-GPU trace | CPU-MoE trace 與 selected-expert GPU offload smoke test |

每次 request 主要記錄：

| 指標 | 用途 |
|---|---|
| TTFT | 第一個 token 的等待時間 |
| total latency | 整體 completion 時間 |
| tok/s | decode throughput |
| major faults | 是否發生磁碟讀取型 page fault |
| minor faults | page table / memory mapping 相關成本 |
| disk read delta | 實際磁碟讀取量 |
| SwapFree / SwapCached delta | 記憶體壓力與 swap 狀態 |
| prefetch MB / prefetch ms | RPP 預取本身的成本 |

### 3.5 RPP-GPU Trace 與下一步快取設計

前面的 RPP prefetch 主要是把 expert weights 讀進 CPU page cache；但若要讓 MoE matmul 更有效利用 GPU，還需要確認 expert weights 能不能被搬到 GPU 並在 GPU 上計算。

因此後續又加入 RPP-GPU trace 實驗，方法如下：

1. 修改 llama.cpp，在 MoE compute node 與 selected-expert copy path 加上 trace hook。
2. 使用 `--ngl 999 --cpu-moe` 作為基底，讓 dense/attention 盡量在 GPU，MoE expert weights 先保留在 CPU mapped memory。
3. 比較不同 `GGML_OP_OFFLOAD_MIN_BATCH`：
   - threshold 32：大型 MoE expert matmul 多數仍在 CPU，沒有 selected-expert H2D copy。
   - threshold 1：大型 MoE expert matmul 幾乎 offload 到 CUDA0，並觸發 selected expert H2D copy。
4. 記錄每次 H2D copy 的 expert tensor、selected expert ids、copy ranges、payload bytes 與 enqueue time。

這一步的目的不是直接追求更快，而是確認 llama.cpp 是否已經存在「只搬本次 router 選到的 experts」這條路徑。若此路徑可行，下一步就能在其上加入 GPU expert cache：

```text
RPP 預測 upcoming experts
-> CPU page cache / host staging
-> async copy 到 GPU VRAM expert cache
-> MoE layer 執行時優先 cache hit
-> cache miss 才 fallback 或同步補搬
```

因此整個方法從 CPU page fault prefetch，逐步延伸到 GPU expert cache；前者驗證 RPP 是否能抓到有用 expert，後者則是讓這些預測真正轉換成 GPU 推論加速。

### 3.6 CPU-GPU 協同推論與 RPP-GPU 方法

本研究中我負責的下半部系統設計，重點是把前面 RPP/FTO 產生的 expert access prediction 轉換成真正能在 GPU 推論路徑上使用的資料搬移與快取策略。核心問題是：在本機 RTX 4060 Laptop GPU 只有 8GiB VRAM 的情況下，Qwen3.6-35B-A3B 的完整 MoE expert weights 無法一次放進 VRAM；因此不能採用「模型全載入 GPU」的方式，而必須讓 CPU system memory / SSD / GPU VRAM 形成分層式 memory hierarchy。

整體設計目標如下：

```text
CPU / SSD side:
  GGUF mmap file
  -> Linux page cache
  -> host memory / staging buffer
  -> RPP predictor or oracle hint producer
  -> expert cache manager

GPU side:
  dense layers / attention / router / selected expert matmul
  -> GPU expert cache hit 時直接使用 VRAM copy
  -> cache miss 時同步或非同步補搬 selected experts
```

與單純 CPU page-cache prefetch 不同，RPP-GPU 的目標不是只把 expert weights 從 SSD 預先讀進 DRAM，而是進一步把 upcoming experts 搬進 GPU VRAM，使後續 MoE layer 的 large matrix multiplication 能在 GPU 上執行，並且盡量避免每次 layer 都重新做 host-to-device transfer。

#### 3.6.1 計算與資料位置切分

實驗中採用 `--ngl 999 --cpu-moe` 作為 RPP-GPU 的基底。這個設定的含義是：dense/attention/non-MoE weights 盡量 offload 到 GPU，而 MoE expert weights 保留在 CPU mapped memory。這樣做的原因有三點：

1. 8GiB VRAM 無法容納完整 MoE expert weights。
2. non-MoE dense/attention layers 較適合長駐 GPU，因為它們每層都會被使用。
3. MoE experts 只有 router 選到的少數 experts 會被使用，適合做 demand paging / cache / prefetch。

因此目前實際計算配置如下：

| component | 實際位置 | 原因 |
|---|---|---|
| tokenization / experiment orchestration | CPU | llama-server API 與 Python runner 控制 |
| RPP predictor / oracle hint generation | CPU / offline | 小型預測模型或 trace parser，不需要佔用 GPU |
| GGUF expert weights 原始資料 | SSD + mmap + CPU page cache | 模型大於 VRAM，expert weight 需以 mmap 方式被動載入 |
| dense / attention / norm | GPU | 計算量大且每層固定使用，適合 offload |
| native router / top-k | 目前保留在 GPU | router 本身很小，放 CPU 會造成 activation 往返，不一定划算 |
| MoE selected expert matmul | GPU | 大型 `MUL_MAT_ID` 是主要計算，應盡量在 CUDA 執行 |
| expert cache manager | 設計上在 CPU control path | 負責決定哪些 expert 要 prefetch / evict / 補搬 |

一開始我們曾討論是否要把 router 強制放在 CPU，讓 CPU 負責所有小型決策、GPU 專心做大型 matmul。但從目前 llama.cpp 的實際行為來看，router/top-k 在 CUDA 上的成本很小；若強制移到 CPU，反而需要在每層把 activation 從 GPU 搬回 CPU，再把 selected expert ids 或中間資料送回 GPU，可能製造額外同步點。因此目前方法採取較保守的切分：RPP predictor 可以在 CPU 背景跑，但主模型原生 router 仍保留在 GPU，用來產生真正正確的 expert selection。

#### 3.6.2 selected-expert GPU offload path

llama.cpp 已經存在一條很重要的 selected-expert copy path。當 MoE expert weight 位於 CPU host buffer，而某個 `MUL_MAT_ID` 被 backend scheduler 指派到 CUDA backend 執行時，llama.cpp 不會把整個 expert tensor 搬到 GPU；它會先讀取 router/top-k 產生的 expert ids，找出本次實際用到的 experts，再只搬這些 selected experts 的連續 byte ranges。

這條路徑可表示為：

```text
GPU router/top-k computes selected expert ids
-> ggml backend scheduler sees host expert weight + CUDA MUL_MAT_ID
-> read selected expert ids back for scheduling
-> merge consecutive expert ids into ranges
-> enqueue H2D copies for selected expert ranges
-> CUDA executes MUL_MAT_ID using copied selected experts
```

為了確認這條路徑是否真的發生，我在 llama.cpp 的 `ggml-backend.cpp` 加入 MoE trace hook。每次 selected-expert copy 發生時會記錄：

| trace field | 含義 |
|---|---|
| `node_name` | 例如 `ffn_moe_gate-12`、`ffn_moe_up-12`、`ffn_moe_down-12` |
| `input_name` | 對應 GGUF expert tensor，例如 `blk.12.ffn_gate_exps.weight` |
| `selected_ids` | 本次 router/top-k 真正選到的 expert ids |
| `ranges` | 合併後的連續 expert id ranges |
| `expert_size_bytes` | 單一 expert 在該 tensor 中的 bytes |
| `copied_payload_bytes` | 不含 padding 的 H2D payload |
| `copied_bytes_with_padding` | 實際 enqueue 的 bytes |
| `enqueue_us_total` | 呼叫 async copy API 的 enqueue time |
| `split_backend` | 該 compute split 實際執行 backend，例如 `CUDA0` |

這個 trace 不改變推論結果，也不改變 scheduler 決策；它只用來觀測目前系統是否真的把 selected expert matmul 放到 GPU，以及每次搬移多少資料。

#### 3.6.3 offload threshold 控制

在 llama.cpp CUDA backend 中，host-weight operation 是否被 offload 到 GPU 會受到 `GGML_OP_OFFLOAD_MIN_BATCH` 影響。預設值為 32，代表 batch size 太小的 host-weight op 不會被 CUDA backend 接手。對 token generation 這種 micro-batch 場景而言，這會導致 MoE expert `MUL_MAT_ID` 留在 CPU 執行。

因此本方法比較了兩種設定：

```text
GGML_OP_OFFLOAD_MIN_BATCH=32
  -> router/top-k 可在 CUDA0
  -> 大型 MoE MUL_MAT_ID 多數仍在 CPU
  -> selected-expert H2D copy events = 0

GGML_OP_OFFLOAD_MIN_BATCH=1
  -> 大型 MoE MUL_MAT_ID 幾乎全部被 offload 到 CUDA0
  -> selected-expert H2D copy path 被觸發
  -> 可觀測每層 selected experts 的 H2D payload
```

這一步的目的不是直接得到最快 latency，而是建立後續 GPU expert cache 的實驗基底。若 MoE expert matmul 仍在 CPU，RPP-GPU cache 就沒有意義；只有當 expert matmul 能在 GPU 執行時，提前把 expert 搬到 VRAM 才可能改善等待時間。

#### 3.6.4 ubatch-aware trace 與 prediction order

llama.cpp server 不是只把一個 request 當成一個完整 batch 直接送進模型，而是會把 logical batch 切成多個 `llama_ubatch`。在 continuous batching 或長 prompt prefill/decode 混合時，真正的 compute order 是 ubatch order，而不是單純的 prompt order 或 token order。

因此 RPP-GPU 的 prefetch 不能只說「這個 request 會用到哪些 experts」，而是要知道：

```text
decode call k
  ubatch 0 uses experts A
  ubatch 1 uses experts B
  ubatch 2 uses experts C
  ...
```

為了讓後續 prefetch 能按照真正執行順序排程，我在 `llama_context::decode()` 周圍加入 ubatch tracing scope，讓每個 MoE trace event 都帶上：

| field | 含義 |
|---|---|
| `llama_decode_index` | 第幾次 llama decode call |
| `llama_ubatch_index` | 該 decode call 中第幾個 ubatch |
| `llama_ubatch_n_tokens` | 這個 ubatch 內 token 數 |
| `llama_ubatch_n_seqs` | 這個 ubatch 內 sequence group 數 |
| `llama_ubatch_pos_min/max` | token position 範圍 |
| `llama_ubatch_seq_id_first` | 第一個 sequence id |

有了這些欄位後，RPP 或 oracle hint 不再只是 request-level prediction，而是可以形成 ubatch-level prefetch schedule：

```text
ubatch i predicted experts
-> prefetch layer l + depth d 的 experts
-> GPU compute 到 layer l 時檢查 cache hit / miss
```

正式 trace 中 60 requests 共記錄到 1980 個 true ubatches，沒有任何 inferred ubatch，代表後續分析已經使用真實 ubatch metadata。

#### 3.6.5 Offline Oracle RPP：100% prediction upper bound

在真正接入 runtime RPP predictor 之前，我先建立 offline oracle 實驗，用主模型 router 實際選到的 experts 當作 RPP 的預測結果。這相當於假設 RPP prediction accuracy = 100%，用來回答一個更基礎的問題：

```text
如果 RPP 完全預測正確，
GPU expert cache 理論上最多能少搬多少 H2D payload？
```

處理流程如下：

```text
1. 跑 --ngl 999 --cpu-moe + GGML_OP_OFFLOAD_MIN_BATCH=1
2. llama.cpp trace 每次 selected-expert copy
3. 解析 trace 中的 selected_ids / layer / tensor kind / ubatch metadata
4. 將每次 selected-expert copy event 轉成 oracle_hint
5. 依照 trace 實際執行順序模擬 GPU expert cache
6. 比較 baseline on-demand copy 與不同 cache capacity 下的 miss payload
```

每筆 oracle hint 代表一次 selected-expert demand：

```text
{
  ubatch id,
  layer id,
  tensor kind: gate / up / down,
  selected expert ids,
  expert size,
  H2D payload bytes,
  original copy ranges
}
```

這個 oracle 實驗目前仍是離線分析，不會改變 llama.cpp runtime，也不會真的減少 latency。它的價值是建立 RPP-GPU 的理論上限：如果 100% 預測都無法帶來足夠 cache hit，那就不值得繼續實作 runtime prefetch；如果 oracle 顯示 cache hit 很高，才代表後續實作 GPU expert cache 有潛在收益。

#### 3.6.6 GPU expert cache 模擬

根據 oracle hints，我實作了 LRU-style 的 GPU expert cache simulation。cache item 的 key 定義為：

```text
(layer id, tensor kind, expert id)
```

例如：

```text
(12, "gate", 87)
(12, "up",   87)
(12, "down", 87)
```

這三個會被視為不同 cache entries，因為它們對應不同 expert tensor 與不同 byte range。模擬時依照 ubatch trace order 逐筆讀取 oracle hints：

```text
for each oracle hint in execution order:
    for each selected expert:
        if expert entry in GPU cache:
            count as cache hit
        else:
            count as cache miss
            add expert bytes to H2D miss payload
            if cache full:
                evict least-recently-used expert entries
```

比較對象如下：

| 方法 | 含義 |
|---|---|
| baseline no cache | 現在 llama.cpp selected-expert offload，每次用到就 H2D copy |
| RPP 100% + 1GB cache | 完美預測，但 VRAM expert cache 只有 1GB |
| RPP 100% + 2GB cache | 完美預測，2GB cache |
| RPP 100% + 4GB cache | 完美預測，4GB cache |
| RPP 100% + 6GB cache | 完美預測，6GB cache |

正式 oracle 使用 20 個 prompts，每個 prompt repeat 3 次，共 60 requests。下表中的 GB 是 60 requests 的累計 H2D payload，不是單次 request；最後一欄另外列出平均每個 request 的 miss payload。

| cache | hit rate | total H2D miss GB | total saved GB | reduction | avg miss GB/request |
|---:|---:|---:|---:|---:|---:|
| baseline | 0.0% | 1572.8 | 0.0 | 0.0% | 26.21 |
| 1GB | 30.5% | 1093.7 | 479.2 | 30.5% | 18.23 |
| 2GB | 43.8% | 884.1 | 688.7 | 43.8% | 14.73 |
| 4GB | 63.3% | 577.7 | 995.1 | 63.3% | 9.63 |
| 6GB | 75.5% | 385.8 | 1187.0 | 75.5% | 6.43 |

![RPP 100% oracle H2D payload per request](experience/RPP-GPU/results/figures/oracle_vs_baseline_h2d_per_request_payload.png)

![RPP 100% oracle expert cache hit rate](experience/RPP-GPU/results/figures/oracle_vs_baseline_cache_hit_rate.png)

這個結果顯示，RPP 100% 本身不是直接讓 H2D payload 變少；真正讓 H2D payload 下降的是「GPU expert cache」。RPP 的角色是提供未來會用到哪些 experts 的資訊，讓 cache manager 能提前搬入、保留或淘汰 expert entries。若 cache 太小，例如 256MB 或 512MB，幾乎沒有 hit；當 cache 達到 4GB 到 6GB 時，hit rate 才變得明顯。

#### 3.6.7 預計 runtime design：prefetch / compute overlap

根據 trace 與 oracle 分析，後續真正要實作的 runtime pipeline 會是：

```text
1. CPU 端 RPP 根據目前 context 預測 upcoming ubatch 的 routing path。

2. cache manager 將 predicted experts 轉換成 expert cache keys：
   (layer, tensor kind, expert id)

3. CPU background worker 檢查哪些 predicted experts 不在 GPU cache。

4. 對 missing predicted experts：
   SSD / mmap page cache -> host memory
   host memory -> cudaMemcpyAsync -> GPU expert cache slot

5. GPU 主 compute stream 正常執行 dense / attention / router。

6. 到達 MoE layer 時，使用 true router selected experts 檢查 cache：
   hit  -> 直接使用 cache slot
   miss -> 同步補搬 missing true experts

7. expert matmul 完成後，GPU event 通知 cache manager：
   該 layer 使用中的 cache slots 可以釋放或降權。

8. cache manager 根據 LRU / frequency / FTO 分數決定 eviction。
```

這裡的 overlap 來自兩條相對獨立的路徑：

```text
GPU compute path:
  layer L attention / norm / router / expert matmul

CPU + copy path:
  RPP predicts layer L+1 ... L+d
  -> page cache / host staging
  -> async H2D prefetch into expert cache
```

理想情況下，CPU 與 copy stream 在 GPU 計算 layer L 的同時，已經把 layer L+1 到 L+d 可能會用到的 experts 搬入 VRAM。當 GPU 真正走到下一層 MoE 時，如果 true router selected experts 已在 cache，就能避免同步 H2D stall；若只有部分命中，則只需補搬 missing experts。

#### 3.6.8 目前方法完成度與限制

目前已完成的部分包含：

1. 確認 `--cpu-moe` 下 MoE expert weights 可保留在 CPU mapped memory。
2. 確認降低 `GGML_OP_OFFLOAD_MIN_BATCH` 後，selected expert `MUL_MAT_ID` 可在 GPU 上執行。
3. 實作 selected-expert H2D copy trace。
4. 實作 ubatch-aware trace metadata。
5. 實作 offline oracle hints，將真實 router path 轉成 100% accurate RPP hints。
6. 實作 GPU expert cache simulation，估算不同 cache 容量下的 H2D payload upper bound。
7. 產生 baseline vs RPP 100% oracle 的圖表與比較報告。

目前尚未完成的部分包含：

1. 還沒有在 llama.cpp runtime 中實作持久化 GPU expert cache。
2. 還沒有讓 `MUL_MAT_ID` 直接讀取自訂 cache slot。
3. 還沒有實作 background CUDA copy stream 與 compute stream 的事件同步。
4. 還沒有把真實 RPP predictor 接進 server decode loop。
5. 目前 oracle 結果只能代表理論上限，不能直接視為實測 latency improvement。

因此目前方法的定位是：先證明 selected-expert GPU execution path 存在，並用 oracle 分析確認「如果 RPP 預測足夠準且 VRAM cache 足夠大，H2D payload 有明顯下降空間」。下一步才是把這個 upper bound 轉成實際 runtime cache / prefetch / overlap implementation。

### 3.7 實驗設計與控制變因

為了避免把不同因素混在一起，本研究採用 phased experiment design。每個 phase 只回答一個主要問題，並保留對照組。

| phase | 目標 | 方法 | 對照組 | 主要輸出 |
|---|---|---|---|---|
| Phase 0 | 建立可重現的本機環境 | 下載模型、確認磁碟/RAM/GPU、建立 prompt set | 無 | model path、server config、prompt database |
| Phase 1 | 找出 CPU/GPU baseline | CPU-only、`--ngl`、`--cpu-moe`、`--n-cpu-moe` | CPU cold/warm、不同 GPU offload | latency、tok/s、page faults、OOM 狀態 |
| Phase 2 | 測試 CPU page-cache RPP prefetch | first-token RPP、expert range prefetch、top-k sweep | two-step no-RPP | prefetch MB、major faults、TTFT、total latency |
| Phase 3 | 建立 RPP-GPU 的可觀測性 | selected-expert H2D trace、ubatch metadata、offline oracle | threshold 32 vs 1、baseline no cache | H2D payload、copy ranges、cache upper bound |
| Phase 4 | 驗證 runtime GPU expert cache | demand-only cache capacity sweep | 0MB cache | actual H2D、D2D、latency、cache hit/miss |
| Phase 5 | 驗證 real-RPP hints 是否有淨收益 | rank top-k 與 FTO admission smoke | demand-only 4GB cache | extra H2D、RPP latency、hint hit/miss |

主要控制變因如下：

1. Prompt set 固定為 `experience/RPP/analyze/prompts.jsonl` 中的 20 個 prompts，涵蓋 text continuation、multiple choice、math reasoning、code generation、commonsense。
2. 大多數 formal comparison 使用 20 prompts × 3 repeat，共 60 requests；smoke test 則使用較少 prompts，用於確認 runtime path 是否正常。
3. Generation 參數以 `n_predict=32`、`temperature=0.0`、`ctx=2048`、`threads=8` 為主，降低 sampling 隨機性。
4. CPU page-cache 實驗會區分 cold/warm 或 no-RPP/RPP matched comparison，避免把自然 cache warming 誤認為 RPP 效果。
5. Runtime GPU cache formal sweep 目前採用 request-level server restart 或固定 scenario order；因此它較像單 request latency / H2D 實驗，尚未完整代表 long-running continuous batching server。
6. 所有 RPP-GPU 數據都要和 demand-only cache 分開比較，因為 cache 本身就會帶來大量 reuse，不能把這部分算成 RPP prediction 的貢獻。

### 3.8 評估指標

本研究同時記錄模型效能、OS IO 與 GPU transfer 指標。各指標意義如下：

| 指標 | 層級 | 解釋 |
|---|---|---|
| TTFT | serving latency | request 送出到第一個 token 產生的時間 |
| total latency | serving latency | 整個 completion 完成時間 |
| tok/s | serving throughput | decode throughput，越高越好 |
| major faults | OS / page cache | 需要從 disk 讀 page 的 page fault 次數，可視為 mmap IO 壓力 proxy |
| minor faults | OS / memory mapping | page table 或已在 memory 中的 mapping fault，通常不代表 disk IO |
| disk read delta | storage | request 前後 block device read bytes 差異 |
| swap delta | memory pressure | 判斷是否因 RAM 不足造成 swap activity |
| prefetch MB / ms | RPP CPU prefetch | RPP 預取量與同步預取成本 |
| H2D payload | GPU transfer | expert weights 從 CPU host 搬到 GPU 的有效 payload |
| actual H2D | runtime GPU cache | 實際發生的 host-to-device transfer，包含 cache miss 與 hint admission |
| D2D | runtime GPU cache | GPU cache hit 後仍需搬到 compute staging buffer 的 device copy |
| cache hit rate | GPU cache | selected expert demand 命中持久化 cache 的比例 |
| extra H2D | RPP hint | RPP hint 相對 demand-only 多搬或少搬的 H2D；小於 0 才代表 RPP admission 真正省 transfer |

判斷一個 RPP-GPU policy 是否真的有效，需要同時滿足兩個條件：第一，total latency 或 TTFT 不能劣化；第二，total H2D 或 actual H2D 不能高於 matched demand-only baseline。若只降低部分 page faults，但 prefetch latency 更高，或只增加 hint hits 但 total H2D 也增加，都不能算成完整的 end-to-end 優化。

## 四、實驗結果

### 4.0 實驗矩陣總覽

下表整理每一組實驗實際回答的問題，以及目前得到的主要結論。

| 實驗組 | 驗證問題 | 最重要對照 | 主要結論 |
|---|---|---|---|
| CPU cold/warm baseline | CPU mmap 是否可靠 | cold vs warm | 21GB 模型大於 RAM，warm cache 幾乎無法改善，CPU-only 明顯 IO-bound |
| GPU `--ngl` sweep | VRAM 能放多少 dense layer | ngl5/10/15/20 | ngl10 可跑，ngl15/20 OOM，純增加 ngl 受 8GiB VRAM 限制 |
| `--cpu-moe` / `--n-cpu-moe` | MoE weights 留在 CPU 是否更適合 8GiB VRAM | ngl10、cpu-moe、n-cpu-moe32 | `baseline-gpu-ncpu-moe-32` 是目前最佳 latency baseline |
| first-token RPP prefetch | RPP page-cache prefetch 是否降低 faults | two-step no-RPP | faults 有下降空間，但同步 prefetch 造成 total latency 劣化 |
| top-k sweep | top-k 越大是否越好 | top2/top4/top8/no-RPP | top-k 越大 prefetch 越多，不保證 latency 更好 |
| selected-expert trace | GPU selected MoE matmul 是否可觀測 | offload threshold 32 vs 1 | threshold 1 會觸發大量 H2D，證明需要 cache |
| offline oracle | 100% prediction 上限多大 | no-cache baseline | 4GB/6GB cache 理論上可大幅降低 H2D payload |
| offline real-RPP | real-RPP 距離 oracle 多遠 | oracle、demand-only simulation | first-token RPP recall 有限，不能直接期待接 runtime 就加速 |
| runtime demand-only cache | cache 本身是否有效 | 0MB cache | 4GB cache 已能實際降低 actual H2D 與 latency |
| runtime RPP hint smoke | RPP/FTO 是否能在 runtime 上勝過 demand-only | 4GB demand-only | naive top-k 會增加 H2D，FTO admit-1 最接近可用但仍未勝過 demand-only |

### 4.1 RPP Predictor 訓練結果

正式訓練比較了 `d_model=32` 與 `d_model=64`。`d_model=64` 明顯較好。

| 模型 | val token_recall@8 | test token_recall@8 | val batch_acc@8 | test batch_acc@8 |
|---|---:|---:|---:|---:|
| RPP d32 | 0.666 | 0.667 | 0.780 | 0.782 |
| RPP d64 | 0.712 | 0.714 | 0.848 | 0.847 |
| Tiny 訓練 | 0.422 | 0.409 | 0.280 | 0.265 |

結果顯示完整資料集與較大的 hidden dimension 能有效提升 routing 預測能力。`d64` 的 test token recall@8 約 0.714，已達到可用於 prefetch 實驗的水準。

### 4.2 CPU Baseline

CPU-only mmap baseline 結果如下：

| scenario | mean TTFT (s) | mean total (s) | tok/s | major faults |
|---|---:|---:|---:|---:|
| baseline-cpu-cold | 5.385 | 32.393 | 0.962 | 624,187 |
| baseline-cpu-warm | 5.472 | 32.398 | 0.959 | 625,713 |

CPU warm workload 幾乎沒有改善 latency，也沒有降低 major faults。這表示 21GB GGUF 在 14GiB RAM 的環境下無法穩定保留於 page cache，CPU-only 推論明顯受記憶體壓力與 IO 限制。

### 4.3 GPU Baseline 與 VRAM 上限

在 8GiB VRAM 下，一般 layer offload 的可行範圍有限：

| scenario | 結果 | mean total (s) | tok/s | major faults | 備註 |
|---|---|---:|---:|---:|---|
| baseline-gpu-ngl5 | success | 5.550 | 5.766 | 120,536 | smoke |
| baseline-gpu-ngl10 | success | 5.732 | 5.583 | 151,425 | smoke |
| baseline-gpu-ngl15 | failed | - | - | - | compute buffer OOM |
| baseline-gpu-ngl20 | failed | - | - | - | CUDA model buffer OOM |
| baseline-gpu-cpu-moe | success | 4.141 | 7.728 | 107,395 | smoke |

正式 20 prompts 實驗中，GPU baseline 相對 CPU baseline 有明顯改善：

| scenario | mean TTFT (s) | mean total (s) | tok/s | major faults |
|---|---:|---:|---:|---:|
| baseline-gpu-ngl10 | 1.262 | 3.105 | 10.179 | 10,612 |
| baseline-gpu-cpu-moe | 1.416 | 2.865 | 12.041 | 17,782 |

![GPU baseline total latency](experience/RPP/analyze/results/figures/gpu_all_baseline_total_latency.png)

![GPU baseline throughput](experience/RPP/analyze/results/figures/gpu_all_baseline_throughput.png)

相較 CPU cold baseline，GPU baseline 的 total latency 約提升 10x 以上，major faults 降低約 97% 到 98%。這代表 GPU offload 即使無法放完整模型，也能大幅降低推論瓶頸。

### 4.4 Partial MoE Offload

接著測試 `--n-cpu-moe`，也就是讓部分 MoE expert weights 留在 CPU，部分放進 GPU。正式比較如下：

| scenario | mean TTFT (s) | mean total (s) | tok/s | major faults |
|---|---:|---:|---:|---:|
| baseline-gpu-cpu-moe | 1.356 | 2.736 | 12.383 | 16,373 |
| baseline-gpu-ncpu-moe-32 | 0.882 | 1.892 | 17.464 | 8,952 |

`baseline-gpu-ncpu-moe-32` 相對 `--cpu-moe`：

| 指標 | 改善幅度 |
|---|---:|
| total latency | 降低 30.85% |
| TTFT | 降低 34.96% |
| tok/s | 提升 41.03% |
| major faults | 降低 45.32% |

結果顯示在本機 8GiB VRAM 下，`--n-cpu-moe 32` 是比純 `--cpu-moe` 更好的 baseline。它能把 VRAM 使用拉高到約 6.6GiB，同時仍避免 OOM。

![Partial MoE offload total latency](experience/RPP/analyze/results/figures/gpu_moe_baseline_total_latency.png)

### 4.5 RPP Prefetch 實驗

RPP first-token prefetch 的目標是先用 prompt 預測後續會用到的 experts，並對 GGUF byte ranges 做 prefetch。

CPU 對照：

| scenario | total (s) | first token (s) | continuation (s) | RPP ms | prefetch ms | prefetch MB | major faults |
|---|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-cpu | 16.164 | 5.436 | 8.956 | 181.8 | 718.3 | 612 | 210,662 |
| two-step-no-rpp-cpu | 12.666 | 5.454 | 7.212 | 0.0 | 0.0 | 0 | 175,989 |

GPU 對照：

| scenario | total (s) | first token (s) | continuation (s) | RPP ms | prefetch ms | prefetch MB | major faults |
|---|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-gpu | 5.828 | 2.994 | 1.752 | 147.4 | 148.4 | 487 | 105,538 |
| two-step-no-rpp-gpu | 4.991 | 3.060 | 1.931 | 0.0 | 0.0 | 0 | 122,608 |

![RPP GPU vs no-RPP total latency](experience/RPP/analyze/results/figures/phase2_gpu_rpp_vs_no_rpp_total_latency.png)

![RPP GPU vs no-RPP major fault breakdown](experience/RPP/analyze/results/figures/phase2_gpu_rpp_vs_no_rpp_major_fault_breakdown.png)

RPP GPU 相對 no-RPP GPU 的 first token 與 continuation 略有改善，major faults 也從 122,608 降到 105,538。不過因為 tokenize、RPP 計算與 prefetch 本身有額外開銷，總延遲反而從 4.991s 增加到 5.828s。

### 4.6 RPP Top-K Sweep

RPP GPU prefetch 的 top-k sweep 結果如下：

| scenario | total (s) | prefetch MB | major faults | first major | continuation major |
|---|---:|---:|---:|---:|---:|
| top-k 2 | 5.933 | 122 | 88,135 | 62,401 | 25,735 |
| top-k 4 | 6.029 | 243 | 73,776 | 49,982 | 23,794 |
| top-k 8 | 5.828 | 487 | 105,538 | 88,850 | 16,688 |
| no-RPP | 4.991 | 0 | 122,608 | 99,600 | 23,009 |

![RPP GPU top-k sweep total latency](experience/RPP/analyze/results/figures/phase2_gpu_topk_sweep_total_latency.png)

![RPP GPU top-k sweep major faults](experience/RPP/analyze/results/figures/phase2_gpu_topk_sweep_major_faults.png)

top-k 4 的總 major faults 最低，但 total latency 沒有最好；top-k 8 的 continuation faults 最低，但 prefetch MB 最大。這表示 prefetch 範圍不是越大越好，還需要更精準的 cache / async 策略。

### 4.7 RPP-GPU Trace 與 Expert Offload

後續實驗轉向新的 RPP-GPU 架構：不只把 expert page 讀進 CPU page cache，而是希望將 upcoming experts staging 到 host memory，再 async 搬到 GPU VRAM expert cache。

Phase 1 先觀察 llama.cpp 既有 selected-expert copy 行為：

| scenario | offload min batch | total (s) | TTFT (s) | tok/s | major faults | H2D events | H2D payload |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpu-cpu-moe-phase1 | 32 | 5.100 | 2.651 | 6.275 | 144,317 | 0 | 0 MB |
| gpu-cpu-moe-expert-gpu-offload | 1 | 6.478 | 2.708 | 4.940 | 8,116 | 4,077 | 25,581.7 MB |

當 `GGML_OP_OFFLOAD_MIN_BATCH=32` 時，大型 MoE expert `MUL_MAT_ID` 仍主要在 CPU 執行，沒有 selected-expert H2D copy。當 threshold 改為 1 時，MoE expert matmul 幾乎都被 offload 到 CUDA0，但單 request 產生約 25.6GB 的 H2D payload，因此 latency 反而變差。

這個結果很重要：它證明現有 llama.cpp 已有「只搬 selected experts」的 offload path，但目前缺少持久化 GPU expert cache，所以重複搬移成本過高。

### 4.8 Prompt 類型分析

正式 RPP-GPU trace 使用 20 個 prompts，分成 text continuation、math reasoning、code generation、multiple choice、commonsense 五類；每類 4 個 prompts，每個 prompt repeat 3 次，因此每類共 12 requests。依 `task_type` 分組後結果如下：

| task type | requests | total s | TTFT s | tok/s | major faults | read GB/request | H2D GB/request | used experts/request |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text_continuation | 12 | 7.181 | 3.135 | 4.469 | 8,574 | 6.43 | 25.45 | 39,967 |
| code_generation | 12 | 7.585 | 3.388 | 4.224 | 9,108 | 7.13 | 25.98 | 40,797 |
| commonsense | 12 | 7.751 | 3.480 | 4.164 | 9,208 | 6.77 | 25.67 | 40,312 |
| math_reasoning | 12 | 9.191 | 4.968 | 3.485 | 12,362 | 9.72 | 28.33 | 44,491 |
| multiple_choice | 12 | 8.784 | 5.267 | 2.989 | 11,963 | 9.86 | 25.65 | 40,290 |

![Task type total latency](experience/RPP-GPU/results/figures/task_type_total_latency.png)

![Task type H2D payload](experience/RPP-GPU/results/figures/task_type_h2d_payload.png)

![Task type major faults](experience/RPP-GPU/results/figures/task_type_major_faults.png)

這個分組分析顯示不同 prompt 類型確實會造成不同系統行為。`math_reasoning` 的 H2D payload、used experts 與 total latency 都最高；`multiple_choice` 的 TTFT 與 major faults 偏高，但 H2D 平均沒有最高，代表它的瓶頸可能更接近 page cache / disk read 或 prompt/token 行為，而不只是 H2D 搬移。

### 4.9 Offline Real-RPP First-Token 分析

前面的 Phase 2 oracle 使用真實 router selected experts 當作 100% 正確預測；Phase 3A 則改用真實 RPP d64 checkpoint。此實驗仍是 offline 分析，不會改變 runtime latency；它只回答「真實 RPP first-token prediction 對 continuation GPU expert cache 有多少幫助」。

流程如下：

```text
prompt + first generated token
-> 真實 RPP d64 predictor
-> top-k 2 / 4 / 8 predicted experts
-> 只分析 trace 中 pos >= prompt_token_count 的 continuation demands
-> 模擬 demand-only cache 與 real-RPP prefetch cache
```

正式 60 requests 的 continuation-only demand payload 為 1100.7GB。4GB cache 下結果如下：

| 方法 | hit rate | H2D miss GB | reduction | pred recall(payload) | false positive GB |
|---|---:|---:|---:|---:|---:|
| demand-only cache | 68.9% | 342.4 | 68.9% | 0.0% | 0.0 |
| real RPP top-k 2 | 69.4% | 336.5 | 69.4% | 4.1% | 2.8 |
| real RPP top-k 4 | 70.0% | 330.4 | 70.0% | 7.5% | 5.3 |
| real RPP top-k 8 | 71.0% | 319.2 | 71.0% | 13.2% | 11.1 |

![Real RPP H2D miss by top-k and cache](experience/RPP-GPU/results/figures/real_rpp_h2d_miss_by_topk_cache.png)

![Real RPP prediction recall](experience/RPP-GPU/results/figures/real_rpp_prediction_recall_4gb.png)

![Real RPP late payload](experience/RPP-GPU/results/figures/real_rpp_late_payload_4gb.png)

結果顯示，真實 RPP top-k 8 在 4GB cache 下相對 demand-only cache 額外減少約 23.2GB H2D miss payload。這代表 real RPP 有幫助，但幅度遠小於 oracle upper bound。以假設 H2D bandwidth = 12GB/s 估計，top-k 8 平均 prefetch finish time 約 56.9ms，late predicted payload 約 0.4GB；因此目前主要問題不是 top-k 8 傳不完，而是 first-token RPP 對整段 continuation 的 payload recall 只有約 13.2%。

### 4.10 Runtime Demand-only GPU Expert Cache

根據 oracle 與 real-RPP offline 結果，下一步先實作不含 RPP 的 runtime demand-only GPU expert cache，作為後續 RPP prefetch 的基準。這一版的目標不是預測 future experts，而是確認「只要 selected expert 曾經被搬進 VRAM，後續再用到時是否能避免重複 H2D」。

runtime 流程如下：

```text
router/top-k selected experts
-> 查詢 GPU expert cache
-> hit: GPU cache slot -> input_cpy 做 D2D staging
-> miss: CPU mmap weight -> GPU cache slot 做 H2D
         GPU cache slot -> input_cpy 做 D2D staging
-> GPU 執行原本 MUL_MAT_ID
```

此版本是真實 runtime cache，不是離線模擬。需要注意的是，目前 compute graph 仍讀取原本的 `input_cpy`，因此 cache hit 仍然需要 device-to-device staging copy；它能減少 host-to-device payload，但還不是最終的 direct-read cache。

正式實驗使用 20 個 prompts、repeat 3，共 60 requests。此 sweep 比較 0MB、2GB、4GB 三種 runtime cache 容量：

| cache | requests | total s | TTFT s | tok/s | demand MB/request | actual H2D MB/request | H2D reduction | cache hit | D2D MB/request |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0MB | 60 | 7.753 | 3.856 | 4.026 | 26213.6 | 26213.6 | 0.0% | 0.0% | 0.0 |
| 2GB | 60 | 7.308 | 3.929 | 4.316 | 26213.6 | 16415.0 | 37.4% | 37.2% | 26234.6 |
| 4GB | 60 | 7.014 | 3.958 | 4.517 | 26213.6 | 12739.4 | 51.4% | 51.2% | 26234.6 |

![Runtime cache actual H2D](experience/RPP-GPU/results/figures/phase4_runtime_cache_actual_h2d.png)

![Runtime cache hit rate](experience/RPP-GPU/results/figures/phase4_runtime_cache_hit_rate.png)

![Runtime cache latency](experience/RPP-GPU/results/figures/phase4_runtime_cache_latency.png)

![Runtime cache by task H2D](experience/RPP-GPU/results/figures/phase4_runtime_cache_task_h2d.png)

正式結果顯示，2GB runtime cache 將 actual H2D payload 從 26.21GB/request 降到 16.42GB/request，約少 37.4%；4GB cache 進一步降到 12.74GB/request，約少 51.4%。Latency 也從 7.753s 降到 7.014s，吞吐從 4.026 tok/s 提升到 4.517 tok/s。改善幅度小於 H2D reduction，主要原因是目前 hit/miss 之後仍需 D2D staging，且 H2D miss 還沒有和 GPU compute overlap。

依 task type 來看，text/code/commonsense 在 4GB cache 下 hit rate 約 54% 到 58%，math/multiple-choice 約 43% 到 45%。這表示 expert reuse pattern 會受 prompt 類型影響；後續 RPP runtime prefetch 不應只看整體平均，也需要保留 task-level 分析。

### 4.11 Runtime RPP Hint Admission Smoke

在 demand-only GPU expert cache 完成後，下一步先做一個可驗證的 runtime RPP hint admission，而不是直接宣稱完成 async prefetch。這版流程如下：

```text
first-token request
-> Python 端使用 prompt + first token 跑真實 RPP checkpoint
-> 寫出 hint file: layer tensor_kind expert_id
-> continuation request 設定 GGML_MOE_RPP_HINTS
-> C++ selected-expert copy path 將 hinted experts admit 到 GPU cache
-> true router selected experts 仍照原本 demand path 執行
```

這個版本的價值是確認 RPP prediction 已經能進入 C++ runtime GPU expert cache。限制是 hint admission 仍發生在 selected-expert copy path 中，還沒有 background worker，也沒有把 H2D copy 與 GPU compute overlap。因此它可能降低部分 demand H2D，但也可能因 false positives 增加總 H2D。

1 prompt smoke 使用 4GB cache，分別測 naive rank top-k 2 / 4 / 8，以及 FTO-filtered top-8 admission。FTO 版本先從既有 formal trace 建立每層 expert frequency table，再從 RPP top-8 candidates 中挑歷史上較常被用到的 experts。每個設定都有 matched demand-only continuation control：

| variant | RPP ms | hint slots | demand-only s | RPP hint s | latency delta | demand-only total H2D MB | RPP demand H2D MB | RPP hint H2D MB | RPP total H2D MB | total H2D delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FTO top-8/admit-1 | 1148.2 | 40 | 5.684 | 5.575 | -0.109 | 9880.2 | 9794.7 | 180.7 | 9975.4 | +95.2 |
| FTO top-8/admit-2 | 1271.0 | 79 | 5.745 | 6.065 | +0.321 | 9880.2 | 9741.3 | 358.3 | 10099.6 | +219.3 |
| Rank top-2 | 1103.6 | 80 | 5.852 | 5.377 | -0.474 | 9880.2 | 9862.1 | 366.6 | 10228.6 | +348.4 |
| Rank top-4 | 1232.7 | 160 | 5.321 | 6.225 | +0.904 | 9880.2 | 10013.2 | 740.8 | 10753.9 | +873.7 |
| Rank top-8 | 1102.3 | 320 | 5.417 | 6.030 | +0.613 | 9880.2 | 10146.3 | 1531.8 | 11678.2 | +1798.0 |

![Runtime RPP hint total H2D](experience/RPP-GPU/results/figures/phase5_runtime_rpp_hint_total_h2d.png)

![Runtime RPP hint latency](experience/RPP-GPU/results/figures/phase5_runtime_rpp_hint_latency.png)

結果顯示，runtime hint admission 的入口是成功的，因為 trace 已記錄到 `rpp_hint_candidates` 與 `rpp_hint_h2d_payload_bytes`。但 naive rank top-k 2 / 4 / 8 都讓 total H2D 增加，代表「把 RPP 預測到的 experts 全部 admit 進 cache」太粗糙。FTO-filtered admission 明顯較好：FTO top-8/admit-1 將 total H2D delta 從 rank top-2 的 +348MB 降到 +95MB；但它仍未低於 demand-only，因此目前還不能宣稱 RPP runtime 已帶來穩定收益。

因此下一步可以優先針對 FTO top-8/admit-1 做小型 repeat，確認單次 smoke 是否穩定；但真正要產生明顯收益，仍需要把 hint admission 改成 background async prefetch，並讓 CPU/RPP/copy stream 在 GPU 計算前一層時提前工作。

### 4.12 Qwen Online RPP-GPU Formal Sweep

為了避免只停留在 smoke test，最後又把 `rpp_runtime_implementation` 版本接到 Qwen3.6，使用 online RPP sidecar 做正式 sweep。這組實驗和前面 Phase 5 的差別是：Phase 5 是透過 hint file 把 RPP predictions 插進 runtime cache path；Phase 6 則是啟動 persistent RPP sidecar，讓 server 在 decode 過程中直接提交 token context 並收到 per-layer predicted experts。

正式 baseline 也重新跑一次，使用相同 prompt set 與 generation 參數：

```text
20 prompts x repeat 3 = 60 requests / scenario
n_predict = 32
ctx_size = 512
--ngl 999
--cpu-moe
GPU expert cache = 1GB
RPP sidecar = CPU
```

比較場景如下：

| scenario | 說明 |
|---|---|
| `rpp-off-native` | 關閉 RPP runtime/cache，作為最新 native baseline |
| `demand-cache-1g` | 啟用 GPU expert cache/correction，但不使用 RPP prediction |
| `online-rpp-top2` | online RPP sidecar，top-k=2 |
| `online-rpp-top4` | online RPP sidecar，top-k=4 |
| `online-rpp-top8` | online RPP sidecar，top-k=8 |

最新 baseline rerun 結果如下：

| scenario | requests | latency s | tok/s |
|---|---:|---:|---:|
| native baseline rerun | 60 | 6.023 | 11.164 |

Phase 6 formal 結果如下：

| scenario | latency s | tok/s | prediction coverage | RPP hit | ready-hit | total H2D GB/request |
|---|---:|---:|---:|---:|---:|---:|
| rpp-off-native | 6.023 | 11.164 | 0.0% | 0.0% | 0.0% | 0.000 |
| demand-cache-1g | 9.734 | 4.942 | 0.0% | 0.0% | 36.7% | 11.626 |
| online-rpp-top2 | 11.027 | 4.300 | 100.0% | 8.6% | 36.6% | 16.233 |
| online-rpp-top4 | 10.768 | 4.264 | 100.0% | 16.2% | 17.2% | 24.134 |
| online-rpp-top8 | 11.558 | 3.915 | 100.0% | 27.4% | 1.2% | 31.787 |

![Final Phase 6 latency vs baseline](experience/RPP-GPU/results/figures/final_phase6_latency_vs_baseline.png)

![Final Phase 6 throughput vs baseline](experience/RPP-GPU/results/figures/final_phase6_throughput_vs_baseline.png)

![Final Phase 6 total H2D vs baseline](experience/RPP-GPU/results/figures/final_phase6_total_h2d_vs_baseline.png)

![Final Phase 6 hit rate vs ready-hit](experience/RPP-GPU/results/figures/final_phase6_hit_rate_vs_ready_hit.png)

這組正式結果有兩個重點。第一，online RPP path 是真的接通了：top2/top4/top8 的 `prediction coverage` 都是 100%，代表 sidecar prediction 有被 runtime 使用。第二，這條路徑目前沒有帶來 end-to-end 加速。所有 online RPP top-k 都比 demand-only cache 慢，也都比 fresh native baseline 慢。

造成這個結果的原因不是 RPP forward 太慢。sidecar 平均 inference time 只有約 4 到 5ms，遠小於每個 request 的整體 latency。真正的問題是 naive top-k prefetch 會把大量 predicted experts 提前送進 cache，造成額外 H2D 與 cache churn。top-k 從 2 增加到 8 時，RPP hit rate 從 8.6% 增加到 27.4%，但 total H2D 也從 16.23GB/request 增加到 31.79GB/request，最後吞吐反而下降。

為了避免混淆不同階段的結論，最終比較圖將 Phase 4 demand-only cache、Phase 5 hint smoke 與 Phase 6 online RPP formal 分開整理：

![Final Phase 4 cache latency vs baseline](experience/RPP-GPU/results/figures/final_phase4_cache_latency_vs_baseline.png)

![Final Phase 4 cache H2D reduction](experience/RPP-GPU/results/figures/final_phase4_cache_h2d_reduction.png)

![Final Phase 5 hint latency delta](experience/RPP-GPU/results/figures/final_phase5_hint_latency_delta.png)

![Final Phase 5 hint H2D delta](experience/RPP-GPU/results/figures/final_phase5_hint_h2d_delta.png)

因此最後的正式結論是：目前沒有任何 RPP-GPU formal 設定勝過最新 native baseline。Phase 4 顯示 GPU expert cache 本身有 component-level 價值；Phase 5 顯示 FTO/admission 比 naive top-k 更有希望；Phase 6 則證明 online sidecar path 已功能完整，但 naive online RPP top-k prefetch 還不是有效優化。

### 4.13 前期 IO-fine / Gemma 實驗

前期 `experience/GPU/IO-fine` 使用 Gemma 4 26B 進行 GPU fit-target 與 IO-bound 偵測。該實驗顯示在 8GiB VRAM 限制下，即使 GPU peak utilization 可達約 54% 到 87%，仍會出現大量 major faults 與 55GB 到 80GB 級別的 disk read delta，所有記錄皆被標記為 IO-bound。

此結果雖然不是 Qwen3.6 主實驗，但支持後續方向：大模型在記憶體不足的本機環境下，推論效能不只受 compute 影響，IO 與記憶體搬移同樣是主要瓶頸。

## 五、分析

### 5.1 主要瓶頸

CPU-only 結果顯示，Qwen3.6 21GB GGUF 在 14GiB RAM 上無法靠 warm cache 解決 IO 問題。major faults 長期維持在 60 萬級，total latency 約 32s，吞吐不到 1 tok/s。

GPU offload 是最有效的第一層改善。即使 VRAM 不足以放完整模型，`ngl10`、`--cpu-moe`、`--n-cpu-moe 32` 都能大幅降低 latency 與 page faults。

### 5.2 最佳 baseline

目前正式實驗中最好的 baseline 是：

```text
baseline-gpu-ncpu-moe-32
TTFT: 0.882s
total: 1.892s
tok/s: 17.464
major faults: 8,952
```

這組比 `--cpu-moe` 更快，也比 `ngl10` 更有效，是後續 RPP-GPU cache 實驗最合理的對照基準。

### 5.3 RPP 的效果與限制

RPP predictor 本身已經有可用的 routing recall，尤其 d64 版本 test token_recall@8 約 0.714。然而目前 first-token prefetch 實作仍有三個限制：

1. Prefetch 是額外同步成本，尚未充分 overlap 到模型推論時間內。
2. 預取進 CPU page cache 不等於讓 expert matmul 真正在 GPU 上受益。
3. Prefetch top-k 過小會 miss，過大則造成過量 IO。

因此目前 RPP prefetch 能降低部分 page faults，但尚未轉換成穩定的 end-to-end latency 改善。

### 5.4 為什麼需要 GPU Expert Cache

RPP-GPU trace 證明 selected-expert offload path 可行，但 threshold 設為 1 時，每個 request 會產生約 25.6GB H2D payload。這說明單純「用到就搬」不可行，必須加入 cache：

```text
RPP 預測 future experts
-> CPU page cache / host staging
-> cudaMemcpyAsync 到 GPU VRAM expert cache
-> MoE layer 執行時直接 cache hit
```

如果沒有持久化 cache，GPU expert offload 會被 PCIe/H2D 搬移成本吃掉；如果有 cache，RPP 的預測才有機會變成真正的 latency improvement。

oracle cache 模擬也支持這個判斷：在 60 requests 累計下，baseline on-demand copy 需要約 1572.8GB H2D payload；若有 4GB expert cache，miss payload 可降到 577.7GB；若有 6GB expert cache，miss payload 可降到 385.8GB。這代表後續真正值得實作的不是單次同步 prefetch，而是可持續復用的 GPU expert cache。

![RPP 100% oracle H2D reduction](experience/RPP-GPU/results/figures/oracle_vs_baseline_h2d_reduction.png)

不過 real-RPP offline 結果也顯示，oracle upper bound 不能直接當成真實 RPP 的預期收益。first-token RPP top-k 8 對 continuation payload 的 recall 約 13.2%，在 4GB cache 下只比 demand-only cache 額外少搬約 23.2GB。因此進 runtime 時必須先建立 demand-only GPU cache baseline，避免把 cache 本身的重複命中誤算成 RPP 的收益。

目前 Phase 4 已完成 demand-only runtime cache formal sweep。2GB cache 將 actual H2D payload 從 26.21GB/request 降到 16.42GB/request，4GB cache 進一步降到 12.74GB/request；這證明持久化 VRAM cache 在真實 runtime path 中確實能減少重複 H2D 搬移。剩下的主要限制是 hit 時仍需 D2D staging，且 H2D miss 尚未與 GPU compute overlap。

Phase 5 進一步證明 RPP prediction 已可透過 hint file 進入 C++ runtime cache path，但 naive top-k admission 目前沒有改善總 H2D。rank top-k 2 / 4 / 8 在 smoke 中分別額外增加約 0.35GB / 0.87GB / 1.80GB total H2D，表示 real-RPP 的 false positives 或過早 admission 會污染 cache。加入 FTO frequency filter 後，top-8/admit-1 的額外 H2D 降到約 0.095GB，是目前最好的 runtime RPP admission candidate；但它仍未低於 demand-only，因此後續仍需 async prefetch / 更細的 admission policy。

Phase 6 則把 online RPP sidecar 正式接到 Qwen runtime，並重新跑了 fresh native baseline。這一步很重要，因為它不再只是 hint-file smoke，而是完整的 online sidecar formal run。結果顯示，目前沒有任何 RPP-GPU formal setting 勝過 baseline：fresh native baseline 為 6.023s/request，而 demand-cache-1g、online-rpp-top2、online-rpp-top4、online-rpp-top8 分別為 9.734s、11.027s、10.768s、11.558s。雖然 top-k 增加能提升 RPP hit rate，但也同步增加 total H2D payload 與 eviction，導致 end-to-end latency 變差。

### 5.5 為什麼下一步應轉向 Batch-aware / Ubatch-aware 優化

目前 runtime demand-only cache、RPP hint smoke 與 Phase 6 online formal 已經證明三件事：第一，GPU expert cache 本身有 component-level 實際收益；第二，real-RPP hints 若採用 naive admission，容易因 false positives 增加 H2D；第三，即使 online RPP sidecar 功能完整接通，單純 top-k prefetch 仍無法勝過 fresh native baseline。這表示下一步不應只是把 top-k 調大或繼續單 request formal run，而應該把 admission policy 對齊真正的 serving execution order。

單 request 實驗的限制在於 cache reuse 空間較小，而且 RPP prediction 成本、hint loading 成本與 H2D copy 很難被攤平。實際 server 更接近 continuous batching：多個 request 會被切成 ubatches，GPU 依序執行 attention/router/expert matmul，而 CPU 可以在 GPU 計算目前 ubatch 時準備下一個 ubatch 或下一層可能使用的 experts。

因此 batch-aware 優化要回答的新問題是：

1. 多 request / multi-ubatch 情境下，expert access frequency 是否更集中，讓 FTO 更容易選出值得保留的 experts。
2. RPP admission 是否應該限制為「下一個或下幾個 ubatch 即將使用」的 experts，而不是把整個 request 的 top-k 一次塞進 cache。
3. CPU-side RPP 與 H2D worker 是否能在 GPU compute 時間內完成，避免 GPU 等待 prefetch。
4. cache eviction 是否應依照 ubatch distance、layer distance 與 frequency score，而不是單純 LRU。

換句話說，batch-aware 實驗不是另一個附加 benchmark，而是把 RPP 從「提前猜哪些 expert 可能有用」推進到「在正確時間把正確 expert 放到 GPU cache」。這會比單純 top-k sweep 更接近原始目標：讓 CPU/RPP/IO path 與 GPU compute path overlap。

## 六、結論

本系列實驗得到以下結論：

1. Qwen3.6 35B MoE 在本機 14GiB RAM 環境下，CPU-only mmap 推論嚴重受 page fault / IO 限制。
2. GPU partial offload 是目前最有效的改善方法，`baseline-gpu-ncpu-moe-32` 在 8GiB VRAM 下達到最佳表現。
3. RPP d64 predictor 已有不錯的 routing 預測能力，test token_recall@8 約 0.714。
4. RPP CPU/GPU prefetch 能降低部分 page faults，但目前同步 prefetch 成本仍使 end-to-end latency 無法穩定勝過 no-RPP baseline。
5. Offline real-RPP first-token 分析顯示，top-k 8 在 continuation 上有額外收益，但 payload recall 仍有限，必須和 demand-only GPU cache 分開比較。
6. Runtime demand-only GPU expert cache formal sweep 顯示，4GB cache 可減少約 51.4% actual H2D payload，並將平均 latency 從 7.753s 降到 7.014s；但這是 component-level improvement，仍不代表已勝過所有 native baseline。
7. Runtime RPP hint-admission smoke 已確認 RPP hints 能進入 C++ GPU cache path；FTO top-8/admit-1 明顯優於 naive rank top-k，但 total H2D 仍略高於 demand-only。
8. 最新 Phase 6 online RPP-GPU formal sweep 已正式接通 sidecar prediction，top2/top4/top8 皆達到 100% prediction coverage；但沒有任何 online RPP scenario 勝過 fresh native baseline。fresh native baseline 為 6.023s/request，online top2/top4/top8 分別為 11.027s、10.768s、11.558s。
9. 下一步重點應從單 request top-k sweep 轉向 batch-aware / ubatch-aware admission，讓 RPP/FTO 根據即將執行的 ubatch 順序決定 prefetch timing，並嘗試把 H2D copy 與 GPU compute overlap。

## 七、後續工作

建議後續實驗依序進行：

1. 停止單純擴大 top-k，因為 Phase 6 formal 已顯示 top-k 變大會增加 H2D 與 cache churn。
2. 對 FTO top-8/admit-1 做小型 repeat，例如 5 prompts × 3 repeat，確認 smoke 中的 latency 與 H2D 趨勢是否穩定。
3. 建立 batch-aware runner：固定 server 不重啟，測 concurrency 1 / 2 / 4，量測 p50/p95 latency、aggregate tok/s、H2D/request、cache hit rate 與 GPU utilization。
4. 將 RPP hint admission 改成 ubatch-aware：只 admission 距離目前 compute 最近的 upcoming ubatch/layer，並加入 per-layer budget。
5. 改良 RPP hint admission policy：加入 confidence threshold、frequency threshold、time-to-use estimate，避免 top-k false positives 污染 cache。
6. 將 hint admission 從 selected-expert copy path 移到 background async prefetch worker，讓 H2D copy 可以和 GPU compute overlap。
7. 比較 `baseline-gpu-ncpu-moe-32`、fresh native baseline、原生 selected-expert offload、demand-only cache、RPP hint admission、RPP async prefetch cache 六者的 latency、H2D bytes、cache hit rate。
8. 加入 CUDA event 或同步量測，區分 enqueue time 與實際 H2D / D2D copy latency。
9. 規劃 direct-read cache，讓 MoE matmul 直接讀持久化 cache slot，避免 hit 時仍需 D2D staging。

## 八、主要資料來源

| 類別 | 路徑 |
|---|---|
| RPP 總說明 | `experience/RPP/README.md` |
| IO 實驗說明 | `experience/RPP/analyze/README.md` |
| IO 結果總紀錄 | `experience/RPP/analyze/results/result.md` |
| RPP d64 訓練報告 | `experience/RPP/qwen36_rpp/results/rpp_train_d64/REPORT.md` |
| GPU baseline 彙整 | `experience/RPP/analyze/results/gpu_all_baselines_selected.summary.md` |
| RPP vs no-RPP 彙整 | `experience/RPP/analyze/results/phase2_gpu_rpp_vs_no_rpp_selected.phase2.md` |
| Top-K sweep | `experience/RPP/analyze/results/phase2_gpu_topk_sweep_selected.phase2.md` |
| RPP-GPU trace 說明 | `experience/RPP-GPU/README.md` |
| RPP-GPU trace 結果 | `experience/RPP-GPU/results/*.summary.md` |
| RPP 100% oracle 比較 | `experience/RPP-GPU/results/oracle_hints/oracle_vs_baseline_comparison.md` |
| Offline real-RPP 分析 | `experience/RPP-GPU/results/real_rpp_offline/real_rpp_first_token_offline_formal_0625_2251.summary.md` |
| Runtime GPU expert cache formal | `experience/RPP-GPU/results/phase4_runtime_cache_formal_comparison.md` |
| Runtime RPP hint smoke | `experience/RPP-GPU/results/phase5_runtime_rpp_hint_smoke_comparison.md` |
| Qwen online RPP-GPU formal | `experience/RPP-GPU/results/phase6_qwen_online_rpp_gpu_formal_0626_1441.summary.md` |
| Final runtime comparison | `experience/RPP-GPU/results/final_runtime_comparison_0626.md` |
| Fresh baseline rerun | `rpp_runtime_implementation/outputs/qwen36_rpp_gpu/formal/phase6_qwen_native_baseline_rerun_0626_1537.summary.md` |
| Prompt 類型分析 | `experience/RPP-GPU/results/task_type_analysis.md` |
| 分析圖表 | `experience/RPP/analyze/results/figures/`、`experience/RPP-GPU/results/figures/` |
| 前期 IO-fine | `experience/GPU/IO-fine/results.csv` |
