# Report Experiment Code Bundle Manifest

This bundle is a source-code companion for the root `REPORT.md`. It preserves
repo-relative paths for the experiment code needed to reproduce or inspect the
reported experiments. Runtime data, generated results, model weights, and build
outputs are intentionally excluded; see `EXCLUDED.md`.

## Shared Source

- `REPORT.md`: project report for the CPU-GPU mixed RPP prefetch branch.
- `llama.cpp/`: source-side llama.cpp tree, including server and tooling source.
  Build directories, `.git`, virtualenvs, compiled binaries, logs, model weights,
  Torch checkpoints, and NPZ artifacts are excluded.
- `experiments/cpu_gpu_mixed_prefetch/`: local benchmark harness for the mixed
  runtime path: RPP online sidecar, GPU expert-cache prefetch, CPU page-cache
  pretouch, FEO admission, and FEO-aware GPU cache reclaim.
- `experiments/gemma4_global_predictor/`: RPP model, dataset loader, losses,
  metrics, training, evaluation, export, monitoring, Dockerfile, and smoke test.
- `experiments/gemma4_bottleneck/`: prompt generation, router-logit collection,
  NPZ preparation helpers, historical RPP/bottleneck analysis scripts, and
  runner scripts used while developing the reported RPP pipeline.
- `dataset/utils/`: prompt database generation, llama-server label dumping,
  NPZ packing, validation, and monitoring source scripts.
- `vast-ai/`: CUDA training image and Vast.ai runner/monitor scripts for RPP
  training sweeps.

## Experiment Coverage

### CPU-GPU Mixed RPP Prefetch

Primary runners:

- `experiments/cpu_gpu_mixed_prefetch/scripts/run_prefetch_benchmark.py`
- `experiments/cpu_gpu_mixed_prefetch/scripts/analyze_prefetch_benchmark.py`
- `experiments/cpu_gpu_mixed_prefetch/scripts/rpp_sidecar.py`
- `experiments/cpu_gpu_mixed_prefetch/scripts/run_under_memory_limit.sh`

Source dependencies:

- `experiments/cpu_gpu_mixed_prefetch/README.md`
- `experiments/cpu_gpu_mixed_prefetch/prompts/`
- `llama.cpp/common/arg.cpp`
- `llama.cpp/common/common.h`
- `llama.cpp/include/llama.h`
- `llama.cpp/src/llama-rpp-runtime.cpp`
- `llama.cpp/src/llama-rpp-runtime.h`
- `llama.cpp/src/llama-rpp-gpu-cache.cpp`
- `llama.cpp/src/llama-rpp-gpu-cache.h`
- `llama.cpp/src/llama-rpp-gpu-transfer.cpp`
- `llama.cpp/src/llama-rpp-gpu-transfer.h`
- `llama.cpp/src/llama-rpp-prefetch.cpp`
- `llama.cpp/src/llama-rpp-prefetch.h`
- `llama.cpp/tools/server/server-context.cpp`
- `llama.cpp/tools/server/server-rpp-sidecar.cpp`
- `llama.cpp/tools/server/server-rpp-sidecar.h`

External prerequisites:

- Gemma4 26B GGUF model file.
- RPP checkpoint used by the online sidecar.
- Expert page map CSV derived from the GGUF tensor layout.
- CUDA-capable `llama-server` built from this branch for FEO configs.
- Optional cgroup/systemd memory-limit support for memory-pressure tests.
- Generated benchmark outputs under `experiments/cpu_gpu_mixed_prefetch/outputs`
  are local artifacts and are not required source inputs.

### I/O Boundness Characterization

Primary runner:

- `experiments/gemma4_IO_behaviors/cross_memory_boundness/run_cross_memory_boundness.sh`

Source dependencies:

- `experiments/gemma4_IO_behaviors/cross_memory_boundness/utils/run_boundness_case_inside_container.sh`
- `experiments/gemma4_IO_behaviors/cross_memory_boundness/utils/boundness_prompt_runner.py`
- `experiments/gemma4_IO_behaviors/cross_memory_boundness/utils/select_short_prompts.py`
- `experiments/gemma4_IO_behaviors/cross_memory_boundness/utils/summarize_cross_memory_boundness.py`
- `llama.cpp/`

External prerequisites:

- Gemma4 26B GGUF model file.
- Remote Docker-compatible runtime on `nthu-cs`.
- Prompt dataset or prompt-selection input referenced by the runner.
- Generated statistics and figures are not bundled.

### RPP Prediction Quality

Primary runners and utilities:

- `dataset/utils/run_prompts_mtp_10000_docker.sh`
- `dataset/utils/run_prompt1000_pipeline_docker.sh`
- `dataset/utils/rpp_prompt1000_pipeline.py`
- `experiments/gemma4_global_predictor/run_train_rpp_docker.sh`
- `experiments/gemma4_global_predictor/run_decode_layer_precision_docker.sh`
- `experiments/gemma4_global_predictor/train_rpp.py`
- `experiments/gemma4_global_predictor/eval_decode_layer_precision.py`
- `experiments/gemma4_global_predictor/eval_rpp_checkpoint.py`

Source dependencies:

- `experiments/gemma4_global_predictor/dataset.py`
- `experiments/gemma4_global_predictor/model.py`
- `experiments/gemma4_global_predictor/losses.py`
- `experiments/gemma4_global_predictor/metrics.py`
- `experiments/gemma4_global_predictor/grid_search_rpp.py`
- `experiments/gemma4_global_predictor/Dockerfile.rpp_torch`
- `experiments/gemma4_bottleneck/build_global_rpp_prompts.py`
- `experiments/gemma4_bottleneck/materialize_prompt_database.py`
- `experiments/gemma4_bottleneck/prepare_global_predictor_dataset.py`
- `experiments/gemma4_bottleneck/router_prediction_prompts/`
- `llama.cpp/`

External prerequisites:

- Prompt-source datasets for Alpaca, XSum, WMT16, Code Alpaca, and Hendrycks MATH.
- Generated prompt databases under `dataset/prompt*`.
- Router-label dumps and packed NPZ data under `dataset/prompt*/router_label_npz`.
- RPP checkpoints (`*.pt`, `*.ts.pt`) and generated statistics/figures.
- Gemma4 26B GGUF model file.

### RPP Runtime Overlap Feasibility

Primary runners:

- `experiments/gemma4_IO_behaviors/mem8G/run_mem8g_pipeline_calibration.sh`
- `experiments/gemma4_IO_behaviors/mem8G/run_mem8g_runtime_rpp_prefetch.sh`
- `experiments/gemma4_IO_behaviors/mem8G/run_mem8g_runtime_rpp_prefetch_matrix.sh`

Source dependencies:

- `experiments/gemma4_IO_behaviors/mem8G/utils/runtime_rpp_prefetch_probe.py`
- `experiments/gemma4_IO_behaviors/mem8G/utils/runtime_cache_aware_scheduler_probe.py`
- `experiments/gemma4_global_predictor/`
- `llama.cpp/`

External prerequisites:

- Gemma4 26B GGUF model file.
- RPP checkpoint or exported TorchScript model.
- Router-label NPZ data used to drive RPP-side predictions.
- Runtime statistics directories and reports generated on the remote host.

### Offline Expert Admission Study

Primary runner:

- `experiments/gemma4_IO_behaviors/mem8G/run_mem8g_rpp_admission_sim.sh`

Source dependencies:

- `experiments/gemma4_IO_behaviors/mem8G/utils/simulate_rpp_expert_admission.py`
- `experiments/gemma4_global_predictor/dataset.py`
- `experiments/gemma4_global_predictor/eval_rpp_checkpoint.py`
- `experiments/gemma4_global_predictor/model.py`
- `experiments/gemma4_global_predictor/metrics.py`

External prerequisites:

- Packed router-label NPZ dataset.
- RPP checkpoint file.
- Generated simulator CSV/JSON/Markdown reports and plots.

### End-to-End FEO Evaluation

Primary runners:

- `experiments/gemma4_IO_behaviors/mem8G/run_mem8g_live_oracle_window_prefetch.sh`
- `experiments/gemma4_IO_behaviors/mem8G/run_mem8g_live_rpp_scheduler.sh`
- `experiments/gemma4_IO_behaviors/mem8G/run_mem8g_runtime_cache_aware_scheduler.sh`
- `experiments/gemma4_IO_behaviors/mem8G/run_mem8g_cache_aware_scheduler.sh`

Source dependencies:

- `experiments/gemma4_IO_behaviors/mem8G/utils/build_oracle_live_trace.py`
- `experiments/gemma4_IO_behaviors/mem8G/utils/build_truth_prefill_subset.py`
- `experiments/gemma4_IO_behaviors/mem8G/utils/build_short_truth_subset.py`
- `experiments/gemma4_IO_behaviors/mem8G/utils/validate_truth_prefill_subset.py`
- `experiments/gemma4_IO_behaviors/mem8G/utils/live_rpp_fixed_pool_client.py`
- `experiments/gemma4_IO_behaviors/mem8G/utils/analyze_live_rpp_overlap.py`
- `experiments/gemma4_IO_behaviors/mem8G/utils/runtime_rpp_prefetch_probe.py`
- `experiments/gemma4_IO_behaviors/mem8G/utils/runtime_cache_aware_scheduler_probe.py`
- `experiments/gemma4_IO_behaviors/mem8G/utils/simulate_true_logits_rebatch.py`
- `experiments/gemma4_IO_behaviors/mem8G/utils/simulate_cache_aware_request_scheduler.py`
- `experiments/gemma4_global_predictor/`
- `llama.cpp/`

External prerequisites:

- Gemma4 26B GGUF model file.
- RPP checkpoint or TorchScript export.
- Oracle/prefill truth traces derived from router-label NPZ data.
- Prompt files, expected-completion files, and generation settings generated for
  each run.
- Runtime statistics, queue traces, overlap traces, client results, and figures.

## Notes

- Existing experiment behavior is not modified by this bundle.
- Scripts are included as source even when their default arguments point to
  excluded datasets, model weights, statistics directories, or build products.
- `llama.cpp` is kept as a source-side tree per user request; upstream docs,
  examples, and media assets are not individually pruned.
