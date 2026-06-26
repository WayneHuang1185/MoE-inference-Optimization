# RPP with FEO Experiment Code Bundle

This repository is a source-code bundle for the RPP with FEO experiments
described in `REPORT.md`. It is intended to make the experiment code,
runner scripts, and provenance documents easier to inspect without uploading
large generated artifacts.

## What Was Updated

The bundle includes the source-side files needed to understand and rerun the
reported experiments:

- `REPORT.md`: current report text and experiment summary.
- `MANIFEST.md`: mapping from each `REPORT.md` experiment section to its
  primary runners, source dependencies, and external prerequisites.
- `EXCLUDED.md`: explicit list of files and artifacts intentionally left out.
- `llama.cpp/`: source-side llama.cpp tree, including server/tooling source.
- `experiments/gemma4_IO_behaviors/`: I/O boundness, runtime overlap,
  offline admission, and end-to-end FEO/RPP scheduler experiment scripts.
- `experiments/gemma4_global_predictor/`: RPP training, evaluation, model,
  dataset loader, losses, metrics, Dockerfile, and smoke test.
- `experiments/gemma4_bottleneck/`: prompt construction, router-logit dump,
  NPZ preparation, and earlier bottleneck/RPP analysis scripts.
- `experiments/gpu_main_rpp_gpu/`: GPU-main RPP-GPU report, runners, compact
  result summaries, and figures for Qwen3.6 MoE expert-cache experiments.
- `dataset/utils/`: prompt database generation, router-label dumping, NPZ
  packing, validation, and monitoring scripts.
- `vast-ai/`: Vast.ai Docker/runner scripts used for RPP training sweeps.

## What Is Not Included

Large or generated artifacts are not committed to this repository:

- Gemma4/GGUF model weights.
- RPP checkpoints and TorchScript exports, such as `*.pt` and `*.ts.pt`.
- Router-label datasets and packed NPZ files, such as `*.npz`.
- Generated prompt databases under `dataset/prompt*`.
- Runtime statistics, figures, logs, CSV/JSON result files, and build outputs.

These files are excluded because they are large, machine-specific, generated
from experiments, or unsuitable for normal Git history. Some may also exceed
GitHub's file size limits.

## Model Weights And Checkpoints

To rerun the experiments, prepare the missing external artifacts separately:

- Gemma4 26B GGUF model file for llama.cpp inference.
- Built llama.cpp binaries, especially `llama-server` and `llama-tokenize`.
- RPP checkpoint or exported TorchScript model for runtime RPP experiments.
- Router-label NPZ dataset for RPP training/evaluation and offline simulation.

Recommended storage for these artifacts:

- GitHub Releases for checkpoint snapshots.
- Hugging Face Hub or another artifact store for model/checkpoint files.
- A remote workstation path documented in `EXCLUDED.md` or a future
  `CHECKPOINTS.md`, including filenames and SHA256 checksums.

Do not commit large model weights or checkpoints directly into this repo unless
Git LFS and quota limits are intentionally configured.

## Notes

- The experiment scripts still reference the original project-relative paths.
- `MANIFEST.md` is the best starting point for finding the runner for each
  experiment section.
- `EXCLUDED.md` documents every external prerequisite category needed for a full
  rerun.
