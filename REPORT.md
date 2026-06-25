## Method
![image](https://hackmd.io/_uploads/S1VJyQ5ffe.png)

- System structure overview
    我們針對 CPU-only inference 場景設計方法。在這個 setting 下，MoE 模型的 expert weights 主要依賴 host memory 與 page cache；當模型執行到某一個 MoE layer 並第一次需要某些 experts 時，若對應 weights 尚未在 page cache 中，inference 會被 major page fault 阻塞。因此，我們的目標是在不改變模型 routing behavior 的前提下，提前產生 expert predictions 並 prepare 即將被使用的 experts，讓 expert loading 與 model computation overlap。
    
    整體方法由兩個 components 組成：Routing Path Predictor（RPP）與 Frequency Expert Organizer（FEO）。RPP 負責為目前 input context 產生跨 MoE layers 的 expert activation predictions；FEO 則根據 RPP predictions，在 batch level 統計 expert access frequency，並利用 layer 之間的 execution gap 安排 prefetch。

- Routing Path Predicton(RPP)
    Routing Path Predictor（RPP）採用 T5-style encoder-decoder architecture，在第一個 MoE layer 執行之前，產生後續所有 MoE layers 的 routing plan predictions。我們在 decoder 後方接上 $L$ 個 lightweight expert heads，分別對應模型中的 $L$ 個 MoE layers，用來學習每個 layer 實際會選到的 experts。
    
    RPP 的目標是在單次 forward pass 中，產生所有 $L$ 個 MoE layers 的 routing path predictions。假設每個 layer 有 $E$ 個 experts，對於一個 token，我們將其 ground-truth routing path 表示為 binary matrix $r \in \{0, 1\}^{L \times E}$，其中 $r_{l,e}=1$ 表示第 $l$ 個 layer 選中了第 $e$ 個 expert，否則為 $0$。RPP 輸出相同維度的 probability matrix $p \in [0, 1]^{L \times E}$，其中 $p_{l,e}$ 表示第 $l$ 個 layer 使用第 $e$ 個 expert 的 predicted probability。
訓練過程可以視為 multi-label classification，並使用 binary cross-entropy 作為 loss function：
$$
\mathcal{L}
= -\frac{1}{LE}
\sum_{l=1}^{L}
\sum_{e=1}^{E}
\left[
r_{l,e}\log p_{l,e}+ (1-r_{l,e})\log(1-p_{l,e})
\right].
$$
經過 RPP prediction 後，每個 token 都會得到跨 layers 的 expert activation probabilities。這些 token-level predictions 會被送入 FEO，用來在 batch level 估計每個 layer、每個 expert 的 access frequency。

- Frequency Expert Organizer
    Frequency Expert Organizer（FEO）將 RPP 的 token-level predictions 聚合成 batch-level expert demand。給定一個包含 $N$ 個 tokens 的 batch，令 $\hat{r}_{t,l,e} \in \{0,1\}$ 表示 token $t$ 是否被 predicted 會在第 $l$ 個 layer 使用 expert $e$，則 expert count 定義為：

    $$
    \mathrm{count}_{l,e}
    = \sum_{t=1}^{N} \hat{r}_{t,l,e}.
    $$

    FEO 接著定義 density：

    $$
    \mathrm{density}_{l,e}
    = \frac{\mathrm{count}_{l,e}}{N},
    $$


    較高的 density 表示同一個 expert 會被更多 tokens 使用，因此 prefetch 該 expert 的 expected benefit 較高。FEO 只選擇 $\mathrm{density}_{l,e}$ 高於 threshold 的 experts 作為 prefetch candidates；這個 batch-level filtering 可以降低 token-level RPP false positives 對 prefetch decision 的影響，並盡量保留high-frequency experts 在 memory 中。

    FEO 的 prefetch timing 由 layer frontier 決定。因為觀察到，MoE model 處理 batch 為 $B$ 的 tokens 時，順序是按造 MoE layers 依序執行為 $(B,0),(B,1),\ldots,(B,L-1)$。 所以當模型正在執行第 $k$ 層時，FEO 會利用目前 layer 的 computation time，為接下來 $w$ 個 layers 中的 high-frequency experts 提前發出 prefetch request：

    $$
    k < l \leq k+w.
    $$

    在這個 window 內，FEO 依照 batch-level density 選出大於 threshold 的 candidates，並送入 prefetch queue。由於 batch 中的 tokens 會按照 layer-by-layer order 被處理，所以可以很輕易的透過 queue 來維護 prefetch 的 priority，FEO 按照 layer order 將滿足 density condition 的 expert candidates enqueue：也就是越接近目前 execution position 的 future layers 越早進入 queue，較遠的 layers 則延後處理。因此，enqueue order 本身即維持了 layer-aware priority。

整體而言，RPP 提供跨 layers 的 expert demand prediction，而 FEO 根據 batch-level frequency 與 layer timing 將 expert loading 與前面 layers 的 computation overlap，降低後續 layers 的 blocking page fault。


## Experiment
- Experimental Setup
    所有 experiments 皆在 remote Linux workstation 上以 containers 執行。該 workstation 使用 x86_64 Linux，CPU 為 AMD Ryzen 9 9950X 16-Core Processor，提供 16 physical cores、32 hardware threads，host memory 為 59 GiB，swap 為 8 GiB。Runtime evaluation 採用 CPU-only inference，以隔離 host memory pressure 與 page-cache behavior 對 inference latency 的影響；不同 memory-pressure regimes 則由各 experiment 的 Docker memory limit 控制。
    
    實驗的 Target model 為 Gemma4 26B，包含 30 個 MoE layers，每個 MoE layer 有 128 個 experts，且每個 token 在每層 activate 8 個 experts。在 CPU-only inference 下，expert weights 會透過 demand paging 進入 page cache；因此，major page fault 可作為 blocking expert loading 的直接指標。
    
- I/O Boundness Characterization
    此 experiment 使用 30 個 short prompts，每筆輸出最多16個 tokens的限制，在 24G、10G、8G、6G memory budgets 下量測 total phase latency、cgroup major page fault、read volume 與 PSI stalls。
    ![image](https://hackmd.io/_uploads/B1xEHPm9ffe.png)

    結果顯示 24G 幾乎沒有 major page fault、read MB 或 PSI stalls，且 CPU parallelism 維持高水位，因此屬於 compute-bound regime。從 10G 開始，major page fault 與 read MB 大幅上升，PSI I/O 與 memory stalls 也同步出現，而其所造成的後果就是 inference latency 也隨之水漲船高。 於是乎，在後續 FEO experiments 中，我們選擇了 10G、8G、6G 作為 memory-pressure regimes。

- RPP Prediction Quality
    RPP 訓練來源於 5 個 routing-label dataset ， 分別是 3,000  `Alpaca instruction-following prompts`、 2,000 筆  `XSum summarization prompts`、 2,000 `WMT16 machine-translation prompts`、 2,000  `Code Alpaca programming prompts` 與 1,000  `Hendrycks MATH mathematical-reasoning prompts`， 收集完共計 10,000 prompts之後先送進 Gemma4 26B 產生生成回復，限制在10 tokens以內。 對 decode 階段的 tokens 記錄每一層 MoE 選擇專家的分配機率，並選取 top-8 experts 作為訓練的正確解答。最後，依照任務類型與資料來源進行分層抽樣，將收集到的數據切分為 8,000 筆訓練集、400 筆驗證集以及 1,600 筆測試集。
    
    以下是在這 10,000 筆的資料中，decode 階段中 
不同 top-k 對應 model 30 層的 recall rate 
對每個 token 與 layer，令 $G_{t,l}$ 表示 ground-truth router 實際啟用的 expert set，且 $|G_{t,l}|=8$；令 $P^{(k)}_{t,l}$ 表示 RPP 機率最高的 top-$k$ expert candidates。Top-k recall 定義為 $|P^{(k)}_{t,l} \cap G_{t,l}| / \min(k, |G_{t,l}|)$。因此 topk, $k=2,4,6,8$ 衡量 RPP 最有信心的 k 個 candidates 是否落在 true experts 內，top16 則衡量 16 個 candidates 對完整 top-8 expert set 的覆蓋程度。

    ![image](https://hackmd.io/_uploads/H1_nTQqMMe.png)

    | decode top-k | mean recall | expected correct experts |
    |---|---:|---:|
    | top2 | 0.916 | 1.83 |
    | top4 | 0.870 | 3.48 |
    | top6 | 0.820 | 4.92 |
    | top8 | 0.753 | 6.02 |
    | top16 | 0.895 | 7.16 |

    這些結果表示 RPP 的 highest-confidence candidates 在 decode-side labels 上具有高 recall 和 coverage；即使不要求完整 token-level routing path 完全正確，FEO 仍可利用這些 top-k candidates 在 batch level 聚合出有用的 prefetch guidance。
- RPP Runtime Overlap Feasibility
    除了 prediction quality 之外，RPP 還必須滿足 runtime feasibility： RPP prediction 不能成為新的 critical path。為了檢查這點，我們使用 128-token decode run 測試不同 CPU allocation，將 model inference threads 與 RPP sidecar threads 透過 CPU affinity 分開。此 experiment 測量單 prompts 的 128-token decode，並量測每個 prompts' decode step 中 RPP prediction 與 prefetch request 的 p95 latency。

    | CPU allocation | model threads | RPP threads | wall ms/token | token interval p95 | prediction p95 | prefetch p95 | sidecar p95 |
    |---|---:|---:|---:|---:|---:|---:|---:|
    | m30+p2 | 30 | 2 | 1049.804 ms | 1791.114 ms | 19.464 ms | 0.408 ms | 19.872 ms |
    | m31+p1 | 31 | 1 | 922.860 ms | 1651.684 ms | 27.493 ms | 0.451 ms | 27.944 ms |

    結果顯示，即使只配置 1 個 RPP thread，prediction p95 加上 prefetch request p95 仍只有 27.944 ms，遠小於同一 run 的 token interval p95 1651.684 ms；配置 2 個 RPP threads 時，sidecar p95 也只有 19.872 ms。這表示 RPP prediction 在時間尺度上可以放進 model inference 的 decode window 中執行，具備 overlap 的 runtime 條件。
- Offline Expert Admission Study
    此實驗透過 offline admission simulation 將 RPP prediction 轉成 prefetch candidates，估計不同 naive admission policies 對 expert demand loads 的影響。此實驗的目的在於分析 prefetch 能否減少 demand-path loads，以及採用不同策略的 prefetch 會造成多少 waste。

   baseline 不做任何 prefetch，只在真正需要某個 expert 且 cache miss 時才載入，cache replacement 使用 LRU。因此，`demand_lru` 的 demand loads 代表純 demand-loading scheduler 下的 blocking expert loads。對任一 admission strategy $s$，令 $D_0$ 與 $T_0$ 分別為 baseline 的 demand loads 與 total loads，令 $D_s$ 為 strategy $s$ 仍需同步載入的 demand loads，$P_s$ 為提前載入的 prefetch loads，則：
$$
\begin{gathered}
T_s = D_s + P_s, \\
\mathrm{demand\text{-}load\ reduction}_s = \frac{D_0 - D_s}{D_0}, \\
\mathrm{total\text{-}load\ overhead}_s = \frac{T_s - T_0}{T_0}.
\end{gathered}
$$
   ![image](https://hackmd.io/_uploads/H1iDvN9fGe.png)
   Demand-load reduction 衡量 prefetch 消除了多少 blocking loads；total-load overhead 則衡量 prefetch 額外引入多少 load work。理想策略應同時提高 demand-load reduction 並控制 total-load overhead。
   
   其中 `RPP topk` 表示不加額外 budget control，直接把 RPP 對每個 token/layer 給出的 top-k predicted experts 作為 prefetch candidates；因此它代表 naive RPP-prefetch upper setting，能測試 prediction 本身能覆蓋多少 future demand。而`RPP budget_60/120/180` 則是在同一批 RPP candidates 上加入 admission budget，每個 planning unit 最多只允許 60、120 或 180 個 candidates 進入 prefetch set，用來觀察限制 budget 後 demand-load reduction 與 prefetch waste 的取捨。
    | strategy | demand loads | prefetch loads     | total loads | demand-load reduction | total-load overhead |
    |---|---:|---:|---:|---:|---:|
    | demand_lru | 425,589 | 0 | 425,589 | 0.00% | 0.00% |
    | rpp_prefetch_budget_60 | 384,717 | 42,795     | 427,512 | 9.60% | 0.45% |
    | rpp_prefetch_budget_120 | 324,258 | 115,214 | 439,472 | 23.81% | 3.26% |
    | rpp_prefetch_budget_180 | 260,145 | 207,735 | 467,880 | 38.87% | 9.94% |
    | rpp_prefetch_topk | 206,469 | 335,446 | 541,915 | 51.49% | 27.33% |
    
    上圖呈現清楚的 budget tradeoff：admission budget 越大，demand loads 降得越多，但 total loads 也因 prefetch work 增加而上升。Budget 60 只降低 9.60% demand loads，total-load overhead 幾乎為 0.45%；budget 180 則可降低 38.87% demand loads，但 total-load overhead 增加到 9.94%。若完全採用 `rpp_prefetch_topk`，demand-load reduction 可達 51.49%，但 total-load overhead 也上升到 27.33%。這表示 RPP prediction 確實能提前覆蓋 future demand，但 naive admission 會把一部分不必要 experts 也載入，形成 prefetch waste。  
    因此，這個 offline study 提出的核心問題是 admission budget 應如何選擇：budget 太小時，blocking demand loads 降幅有限；budget 太大時，雖然 demand loads 明顯下降，卻可能因 total-load overhead 過高而抵消效益。此 study 本身沒有使用 FEO runtime scheduling，而是作為後續 FEO admission/budget design 的 motivation。
- End-to-End FEO Evaluation
  最後以 final memory-limit sweep 評估 end-to-end runtime effect。我們比較 baseline CPU-only inference 與 FEO-enabled inference；兩者使用相同 prompts、model、memory limit 與 generation parameters。設定使用 50 個 prompts、產生最多 10 tokens、batch size 10、density threshold 0.15、lookahead window 3、每次 planning step 最多 admit 512 個 experts。每次 baseline 與 FEO run 前皆 drop cache。
  我們量測 total wall time 與 cgroup major page faults。 total wall time 衡量 end-to-end inference latency；major page faults 則衡量 inference 當下 expert loading 過程中的 blocking page-cache misses。主要結果如下：
  | memory | baseline wall | FEO wall | wall improve | baseline pgmaj | FEO pgmaj | pgmaj improve |
    |---|---:|---:|---:|---:|---:|---:|
    | 10G | 101.11s | 92.67s | 8.35% | 869,696     | 639,217 | 26.50% |
    | 8G | 190.42s | 184.90s | 2.90% | 1,527,612 | 1,232,579 | 19.31% |
    | 6G | 505.73s | 270.75s | 46.46% | 2,837,646 | 2,096,320 | 26.12% 
    
    FEO 在所有 **memory budgets** 下皆降低 major page faults，表示 layer-aware prefetch 確實減少 blocking page-cache misses。6G 下 wall-time improvement 最大，達到非常驚人的 46.46%，說明當 blocking expert loading 成為主要瓶頸時，提前把 high-density experts 放入 page cache 最有效。8G 下 major page fault reduction 仍有 19.31%，但 wall-time improvement 只有 2.90%，代表在中等 memory pressure 下，prefetch overhead 或 remaining compute cost 仍可能抵消一部分 page-fault reduction。
    
    - limitation of RPP
         而在這個看似很理想實驗的背後，存在著一個問題需要解決，那就是 RPP 的準確率對於 FEO 提升整體效能的影響非常大，我們發現 batch-level metrics 顯示 RPP 可以提供 coarse-grained guidance，但 prefill-side token-level quality，以`token_recall@16` 為例，經過測試 prefill `token_recall@16` 只有 0.429，遠低於 decode phase 的 0.885，這表示若直接把 prefill 階段完全交給 live RPP prediction，prefetch candidates 可能包含大量 false positives，導致漏掉真正需要的 experts，並多載入不需要的 experts。基於這個觀察，此次 runtime evaluation 先採取較保守的 setup：針對 prefill 階段先記錄一次 true router logits，並將這個結果看作是 RPP 預測的結果，而等到了 decode 階段才是真正的將 RPP 帶入預測。 也就是說，這個 end-to-end results 應被視為驗證 FEO 方法的一個 upper bound，而不是完整證明 RPP 在所有 phases 都已能獨立支撐 production-quality prediction。

## Future work
- 提升RPP的準確率，達到除了在 decode 階段有高準確率之外，在 prefill 階段也有，如此方能將 FEO 的方法真正落實到實際 inference 的場景
- 目前沒有改動任何 batch 的順序，只有做 prefetch，也就是說整體的 I/O 數量是沒有下降的，但有了 RPP 的資訊過後，是不是能夠對 batch 做排序，目標做到在一個
batch 中 ， activate 最少的 distinct experts， 透過將有相似 experts 分布的 tokens 聚合成一個 batch，如此再搭配 FEO 就可以達到 I/O 和 inference的
雙重優化
    

    
    