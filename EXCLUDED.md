# Excluded Files And External Prerequisites

This bundle intentionally excludes generated data and machine-specific artifacts.

## Exclusion Rules

Excluded from experiment directories:

- `statistics/`
- `figures/`
- `results/`
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
