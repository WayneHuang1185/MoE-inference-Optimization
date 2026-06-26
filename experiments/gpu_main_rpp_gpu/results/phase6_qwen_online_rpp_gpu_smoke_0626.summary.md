# Phase 6 Qwen Online RPP-GPU Smoke 0626

這次測試目的：確認 `rpp_runtime_implementation` 內的 RPP runtime 版本能不能直接接到本機 Qwen3.6 MoE GGUF，並把真實 RPP sidecar prediction 接進 GPU expert cache / correction path。

## Runtime

- Runtime repo：`/home/hazcashi/lab/rpp_runtime_implementation/llama.cpp`
- Build：`build-rpp-cuda124`
- Binary：`/home/hazcashi/lab/rpp_runtime_implementation/llama.cpp/build-rpp-cuda124/bin/llama-server`
- Model：`/home/hazcashi/lab/model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Page map：`/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/expert_page_map.csv`
- RPP checkpoint：`/home/hazcashi/lab/experience/RPP/qwen36_rpp/results/rpp_train_d64/checkpoint_best.pt`
- RPP config：`/home/hazcashi/lab/experience/RPP/qwen36_rpp/results/rpp_train_d64/config.json`

## Build

```bash
cd /home/hazcashi/lab/rpp_runtime_implementation/llama.cpp
cmake -S . -B build-rpp-cuda124 -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build-rpp-cuda124 --target llama-server test-rpp-args test-rpp-runtime test-rpp-prefetch test-rpp-gpu-cache -j "$(nproc)"
```

## Qwen Expert Page Map

```bash
cd /home/hazcashi/lab/rpp_runtime_implementation
python3 scripts/build_expert_page_map.py \
  --model /home/hazcashi/lab/model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --out outputs/qwen36_rpp_gpu/expert_page_map.csv
```

結果：

| item | value |
|---|---:|
| rows | 30720 |
| layers | 40 |
| experts / layer | 256 |
| components | down, gate, up |
| layer 0 expert 0 total bytes | 1900544 |

## Smoke Command

測試時使用：

```bash
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

Prompt：

```text
Explain RPP in one short sentence.
```

Completion 結果：

| metric | value |
|---|---:|
| tokens evaluated | 9 |
| tokens predicted | 6 |
| prompt ms | 1747.031 |
| predicted ms | 1238.781 |
| predicted tok/s | 4.843 |

## Trace Result

Trace：`/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/rpp_gpu_online_trace.jsonl`

| metric | value |
|---|---:|
| trace events | 200 |
| layers | 0-39 |
| GPU correction success | 200 / 200 |
| prediction found | 200 / 200 |
| prefetched experts | 400 |
| hit experts | 183 |
| missing experts | 1417 |
| evictions | 612 |
| loaded on demand | 919 |
| ready hits | 680 |
| max resident entries | 388 |
| cache slots | 388 |

Sidecar metrics：`/home/hazcashi/lab/rpp_runtime_implementation/outputs/qwen36_rpp_gpu/sidecar_metrics.jsonl`

| metric | value |
|---|---:|
| sidecar requests | 6 |
| last CPU RPP inference ms | 3.345 |

## Interpretation

- Qwen separate expert layout 已經可以被 runtime cache 辨識：`ffn_gate_exps`、`ffn_up_exps`、`ffn_down_exps`。
- Online sidecar path 已經接通，trace 中 `prediction_found = 200 / 200`。
- GPU expert correction path 已經接通，trace 中 `gpu_correction_success = 200 / 200`。
- 這次是 functional smoke，不是正式效能結論；`top-k=2` 只預抓少量 experts，因此大多數 true selected experts 仍需要 on-demand 補搬。
- 256MiB cache 曾經觸發 CUDA illegal memory access；修正後採用 ubatch 結束才釋放保護，1GiB cache smoke 已通過。小 cache 需要另外重跑驗證。
