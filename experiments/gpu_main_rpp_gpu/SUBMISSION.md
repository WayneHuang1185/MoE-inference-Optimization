# GPU-main RPP-GPU Folder Guide

This folder contains the GPU-main part of the MoE inference optimization
project. The focus is Qwen3.6 MoE expert-cache management on the GPU side,
especially host-to-device expert transfer, runtime GPU expert cache, and online
RPP prefetch experiments.

## Folder Layout

| Path | Content |
|---|---|
| `REPORT.md` | HackMD-ready final report for submission. |
| `REPORT_FULL.md` | Longer working report with more detailed provenance. |
| `README.md` | Detailed experiment notes and phase-by-phase records. |
| `figures/` | PNG figures used by the report. |
| `results/` | Compact Markdown/CSV summaries only. Raw traces are excluded. |
| `prompts/prompts.jsonl` | 20-prompt evaluation set used by the local formal runs. |
| `run_phase6_qwen_online_rpp_gpu_formal.py` | Formal online RPP-GPU runner. |
| `run_qwen_online_rpp_gpu_smoke.sh` | Smoke-test entry point for online RPP-GPU. |
| `run_runtime_rpp_hint.py` | Runtime RPP hint experiment runner. |
| `build_oracle_hints.py` | Offline oracle hint builder. |
| `build_real_rpp_offline.py` | Offline real-RPP cache simulation builder. |
| `plot_*.py` | Figure generation scripts. |

## Runtime Code

The implementation is placed under the repository-level `llama.cpp/` directory:

| Path | Content |
|---|---|
| `llama.cpp/src/llama-rpp-runtime.*` | Runtime coordination for RPP-GPU. |
| `llama.cpp/src/llama-rpp-gpu-cache.*` | GPU expert cache data structure and policy. |
| `llama.cpp/src/llama-rpp-gpu-transfer.*` | Expert transfer helpers. |
| `llama.cpp/src/llama-rpp-prefetch.*` | Prefetch/admission path. |
| `llama.cpp/src/llama-rpp-predictor*` | Replay/prediction interfaces. |
| `llama.cpp/tools/server/server-rpp-sidecar.*` | Online sidecar integration. |
| `llama.cpp/tests/test-rpp-*.cpp` | Unit tests for RPP-GPU components. |

## Excluded Artifacts

The following artifacts are intentionally not included in Git:

- GGUF model weights.
- Raw trace JSONL files.
- Server logs.
- Build directories and binaries.
- Python virtual environments.
- Large RPP checkpoints or datasets.

Use the compact summaries in `results/` and figures in `figures/` for report
inspection. Use the runners only after preparing the external model/checkpoint
paths locally.
