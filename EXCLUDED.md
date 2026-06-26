# Excluded Files And External Prerequisites

This bundle intentionally excludes generated data and machine-specific artifacts.

## Exclusion Rules

Excluded from experiment directories:

- `statistics/`
- `figures/`
- `results/`
- `outputs/`
- generated prompt/data payloads under `dataset/prompt*`
- model directories and model weights
- checkpoints and TorchScript exports: `*.pt`, `*.ts.pt`
- packed labels and numeric payloads: `*.npz`
- generated result tables and logs: `*.csv`, generated `*.json`, `*.jsonl`, `*.log`
- Python caches and test caches

Excluded from `llama.cpp/`:

- `.git/`
- `build*/`
- `gguf-py/.venv/`
- compiled objects, binaries, shared libraries, archives, and logs
- model weights and NPZ/Torch artifacts

The `llama.cpp` source tree otherwise remains broad, including upstream docs,
examples, server/web UI source, and media assets.

## External Prerequisites

The following are required to rerun the experiments but are not bundled:

- Gemma4 26B GGUF model files.
- Built llama.cpp binaries, especially `llama-server` and `llama-tokenize`.
- Remote Docker-compatible runtime on `nthu-cs`.
- Prompt-source datasets used to build the 10,000-prompt RPP dataset.
- Generated prompt databases under `dataset/prompt*`.
- Router-label dumps and packed NPZ datasets under `dataset/prompt*/router_label_npz`.
- RPP checkpoints and TorchScript exports.
- Expert page map CSV derived from GGUF tensor offsets.
- Oracle/prefill truth traces generated from router-label NPZ data.
- Per-run prompt directories, expected completions, and generation settings.
- Generated runtime outputs: reports, CSV/JSON summaries, traces, logs, plots,
  and figure files.

## Source Kept Despite Data-Like Path Names

- `dataset/utils/` is included because it contains source scripts for prompt
  collection, label dumping, packing, validation, and monitoring. It does not
  include prompt payloads or NPZ data.
- `experiments/gemma4_bottleneck/router_prediction_prompts/` is included because
  these are small source prompts used by the router-prediction tooling, not
  generated result payloads.
- `experiments/cpu_gpu_mixed_prefetch/prompts/` is included because it contains
  small benchmark prompt lists. The sibling `outputs/` directory is generated
  runtime data and should not be committed by default.

## Branch-Specific External Runtime Artifacts

The CPU-GPU mixed branch also depends on these local artifacts when rerunning
the runtime experiments:

- CUDA build directory: `llama.cpp/build-rpp-cuda118/`
- CPU build directory: `llama.cpp/build-rpp-cpu/`
- RPP sidecar checkpoint: `models/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt`
- Gemma4 GGUF model: `models/workstation_gemma4-26B.gguf`
- Expert page map CSV from the previous GGUF page-map phase.

These paths are documented for reproducibility but are intentionally excluded
from Git.
