# CPU-GPU Mixed RPP Prefetch Code Bundle

This repository is a source-code bundle for the CPU-GPU mixed RPP prefetch
experiments described in `REPORT.md`. It combines two previously separate
directions:

- CPU/page-cache FEO-style optimization: use RPP route hints to prepare GGUF
  expert byte ranges in the CPU page cache.
- GPU expert-cache optimization: use RPP route hints to move selected expert
  slices into GPU VRAM before the true MoE layer needs them.

The repository is meant for source review and reproducible reruns. Large model
weights, checkpoints, router-label datasets, build products, and generated
experiment outputs are intentionally kept out of normal Git history.

## What Was Updated

The bundle includes the source-side files needed to understand and rerun the
reported experiments:

- `REPORT.md`: current method, experiment plan, preliminary observations, and
  future work for the CPU-GPU mixed branch.
- `MANIFEST.md`: mapping from report sections to primary runners, source
  dependencies, and external prerequisites.
- `EXCLUDED.md`: explicit list of files and artifacts intentionally left out.
- `llama.cpp/`: source-side llama.cpp tree with the RPP runtime changes.
- `experiments/cpu_gpu_mixed_prefetch/`: local benchmark harness for comparing
  no predictive prefetch, GPU-only predictive prefetch, naive CPU+GPU mixed
  prefetch, and FEO-filtered CPU+GPU mixed prefetch.
- `experiments/gemma4_IO_behaviors/`: original CPU/page-cache, I/O boundness,
  runtime overlap, offline admission, and end-to-end FEO/RPP scheduler scripts.
- `experiments/gemma4_global_predictor/`: RPP training, evaluation, model,
  dataset loader, losses, metrics, Dockerfile, and smoke test.
- `experiments/gemma4_bottleneck/`: prompt construction, router-logit dump,
  NPZ preparation, and earlier bottleneck/RPP analysis scripts.
- `dataset/utils/`: prompt database generation, router-label dumping, NPZ
  packing, validation, and monitoring scripts.
- `vast-ai/`: Vast.ai Docker/runner scripts used for RPP training sweeps.

## Runtime Features

The `llama.cpp` source in this branch contains the runtime hooks needed for:

- online RPP sidecar calls for decode-token routing prediction;
- true-router correction so RPP does not replace model routing decisions;
- host page-cache prefetch with `willneed` or forced pretouch modes;
- GPU expert cache and GPU correction copy;
- deadline-aware GPU copy queue and multiple copy workers;
- FEO-style prefetch admission based on ubatch route density;
- FEO-aware GPU cache reclaim policy.

Important branch-specific flags include:

```text
--rpp-host-prefetch off|willneed|pretouch
--rpp-gpu-correction on|off
--rpp-gpu-compute on|off
--rpp-gpu-queue-policy fifo|deadline
--rpp-gpu-copy-workers N
--rpp-prefetch-depth D
--rpp-prefetch-top-k K
--rpp-prefetch-admission topk|feo
--rpp-gpu-reclaim-policy lru|feo
```

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

## Build

CPU-only structural build:

```bash
cd /home/jacky20040024/projects/跑gemma/for_repo_branch/MoE-inference-Optimization

cmake -S llama.cpp -B llama.cpp/build-rpp-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_UI=OFF

cmake --build llama.cpp/build-rpp-cpu -j2 \
  --target test-rpp-args test-rpp-runtime
```

CUDA build on the local workstation should use the complete CUDA 11.8 toolkit:

```bash
cd /home/jacky20040024/projects/跑gemma/for_repo_branch/MoE-inference-Optimization

cmake -S llama.cpp -B llama.cpp/build-rpp-cuda118 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-11.8

cmake --build llama.cpp/build-rpp-cuda118 -j2 \
  --target llama-server test-rpp-args test-rpp-runtime
```

The CUDA build can be resumed with the last command if interrupted. This branch
contains a local compatibility patch that disables llama.cpp's PDL path for
CUDA 11.8, because CUDA 11.8 headers do not provide the device APIs used by
that path.

After a successful CUDA build, verify the FEO flags:

```bash
llama.cpp/build-rpp-cuda118/bin/llama-server --help 2>&1 \
  | grep -E "rpp-prefetch-admission|rpp-gpu-reclaim-policy"
```

## Model Weights And Checkpoints

To rerun the experiments, prepare the missing external artifacts separately:

- Gemma4 26B GGUF model file for llama.cpp inference.
- Built llama.cpp binaries, especially `llama-server` and `llama-tokenize`.
- RPP checkpoint or exported TorchScript model for runtime RPP experiments.
- Router-label NPZ dataset for RPP training/evaluation and offline simulation.
- Expert page map derived from the GGUF tensor layout.

Recommended storage for these artifacts:

- GitHub Releases for checkpoint snapshots.
- Hugging Face Hub or another artifact store for model/checkpoint files.
- A remote workstation path documented in `EXCLUDED.md` or a future
  `CHECKPOINTS.md`, including filenames and SHA256 checksums.

Do not commit large model weights or checkpoints directly into this repo unless
Git LFS and quota limits are intentionally configured.

## Smoke Test

The local mixed benchmark harness is documented in:

```text
experiments/cpu_gpu_mixed_prefetch/README.md
```

The most relevant smoke configs are:

```text
rpp_depth_0_c512
rpp_deadline_d1_k2_w2_c512
rpp_mixed_host_d1_k2_w2_c512
rpp_feo_mixed_d1_k2_w2_c512
```

FEO-specific configs require the `llama-server` built from this branch. Older
runtime binaries can be used only for non-FEO smoke tests, because they do not
know `--rpp-prefetch-admission feo` or `--rpp-gpu-reclaim-policy feo`.

## Notes

- The experiment scripts still reference the original project-relative paths.
- `MANIFEST.md` is the best starting point for finding the runner for each
  experiment section.
- `EXCLUDED.md` documents every external prerequisite category needed for a full
  rerun.
- `experiments/cpu_gpu_mixed_prefetch/outputs/` is local generated output and
  should not be committed unless a small result snapshot is intentionally added.
