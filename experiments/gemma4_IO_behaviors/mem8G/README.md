# Gemma4 26B I/O Behavior, 8G Decode

This directory contains the 8G decode-only experiment for expert page-cache
growth and MOE vs non-MOE refault attribution. The tools are a scoped copy of
the mem10G decode sidecars, with outputs redirected under `mem8G`.

The runner uses:

- Docker `--memory=8g --memory-swap=8g`
- CPU-only llama execution with `-ngl 0`
- no GPU passthrough by default; set `DOCKER_GPU_ARGS='--gpus all'` only on a
  Docker host where GPU CDI/runtime is available
- forced llama-server `--no-repack`
- default `N_PREDICT=5`
- default `PROMPT_LIMIT=1`
- `/slots/0?action=erase` before and after every prompt
- `/completion` payload `cache_prompt=false`

## Sync To Remote

```bash
rsync -av experiments/gemma4_IO_behaviors/mem8G/ nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_IO_behaviors/mem8G/
```

## Smoke Run

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && PROMPT_LIMIT=1 N_PREDICT=1 experiments/gemma4_IO_behaviors/mem8G/run_mem8g_decode_expert_cache_fault.sh'
```

## Full Decode Run

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && PROMPT_LIMIT=5 N_PREDICT=5 PERF=0 experiments/gemma4_IO_behaviors/mem8G/run_mem8g_decode_expert_cache_fault.sh'
```

## 10000-Token Resident Timeline

```bash
rsync -av experiments/gemma4_IO_behaviors/mem8G/ nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_IO_behaviors/mem8G/
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && PROMPT_LIMIT=1 N_PREDICT=10000 experiments/gemma4_IO_behaviors/mem8G/run_mem8g_decode_resident_timeline.sh'
```

Smoke check:

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && PROMPT_LIMIT=1 N_PREDICT=8 experiments/gemma4_IO_behaviors/mem8G/run_mem8g_decode_resident_timeline.sh'
```

Timeline statistics stay on the remote host under:

```text
experiments/gemma4_IO_behaviors/mem8G/statistics/resident_timeline_<timestamp>/
```

The two SVG figures are written under:

```text
experiments/gemma4_IO_behaviors/mem8G/figures/resident_timeline_<timestamp>/
```

The per-token CSV columns are `token_index`, `actual_resident_count`,
`predicted_resident_count`, `swapped_in_count`, `swapped_out_count`, and
`elapsed_s`. The predicted count comes from the `expert_capacity` budget model:
`(memory_limit - non_MOE - fixed_runtime_overhead - KV(token)) / avg_expert_size`.

## RPP Expert Admission Simulation

This is the next offline gate before adding a runtime prefetch sidecar. It uses
the trained global RPP checkpoint to admit predicted experts before each decode
token under the mem8G expert-capacity budget, then compares demand-path expert
loads against plain LRU, full RPP top-k prefetch, budgeted RPP prefetch,
thresholded RPP prefetch, and oracle prefetch. It does not measure llama.cpp
latency.

```bash
rsync -av experiments/gemma4_IO_behaviors/mem8G/ nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_IO_behaviors/mem8G/
rsync -av experiments/gemma4_global_predictor/ nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_global_predictor/
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && MAX_SAMPLES=1000 PREDICT_TOPK=8 PREFETCH_BUDGETS=60,120,180 experiments/gemma4_IO_behaviors/mem8G/run_mem8g_rpp_admission_sim.sh'
```

Smoke check:

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && MAX_SAMPLES=8 CACHE_CAPACITY=1156 experiments/gemma4_IO_behaviors/mem8G/run_mem8g_rpp_admission_sim.sh'
```

Outputs stay under:

```text
experiments/gemma4_IO_behaviors/mem8G/statistics/rpp_admission_<timestamp>/
```

Important columns in `summary.csv`:

- `demand_load_reduction_vs_lru`: estimated latency-path load reduction.
- `total_load_reduction_vs_lru`: estimated total I/O change after prefetch
  waste.
- `prefetch_candidate_rate`: requested prefetch candidates divided by true
  expert accesses; use this to compare full top-k vs budgeted strategies.
- `capacity_mean`: dynamic expert-cache capacity from the 8G budget model.

## Runtime RPP fadvise Prefetch

This is the non-invasive runtime bridge from the global RPP predictor into
llama.cpp I/O behavior. The runner starts `llama-server` in the 8G container,
then starts a Python sidecar in the RPP container. The sidecar streams generated
token ids, runs the predictor, maps predicted `(layer, expert)` pairs to GGUF
expert tensor ranges, and calls `posix_fadvise(POSIX_FADV_WILLNEED)` on those
file ranges.

Default runtime settings are conservative:

- `PREFETCH_BUDGET=30`
- `PREFETCH_THRESHOLD=0`
- `PREFETCH_INTERVAL=1`
- `ADVICE_CACHE_TOKENS=4`

Single-case smoke runs:

```bash
rsync -av experiments/gemma4_IO_behaviors/mem8G/ nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_IO_behaviors/mem8G/
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && RUN_TIMESTAMP=rt_none_smoke PORT=8091 N_PREDICT=4 PROMPT_LIMIT=1 PREFETCH_MODE=none experiments/gemma4_IO_behaviors/mem8G/run_mem8g_runtime_rpp_prefetch.sh'
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && RUN_TIMESTAMP=rt_rpp_b30_smoke PORT=8092 N_PREDICT=4 PROMPT_LIMIT=1 PREFETCH_MODE=rpp PREFETCH_BUDGET=30 experiments/gemma4_IO_behaviors/mem8G/run_mem8g_runtime_rpp_prefetch.sh'
```

Comparison matrix:

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && RUN_TIMESTAMP=rt_matrix_n16 REPEATS=1 BASE_PORT=8090 N_PREDICT=16 PROMPT_LIMIT=1 experiments/gemma4_IO_behaviors/mem8G/run_mem8g_runtime_rpp_prefetch_matrix.sh'
```

The matrix runs baseline, budget 30, budget 60, and threshold 0.95, then writes
a combined `REPORT.md` under:

```text
experiments/gemma4_IO_behaviors/mem8G/statistics/runtime_rpp_prefetch_matrix_<timestamp>/
```

## Pipeline Calibration

This is the formal CPU-sidecar calibration pass for runtime RPP prefetch. It
starts fresh mem8G CPU-only `llama-server` containers, measures decode CPU
parallelism with `benchmark_prefill_decode.py`, then runs `none`,
`predict_only`, and `rpp` sidecar modes. The final aggregation writes:

```text
experiments/gemma4_IO_behaviors/mem8G/statistics/pipeline_calibration_<timestamp>/
experiments/gemma4_IO_behaviors/mem8G/figures/pipeline_calibration_<timestamp>/
```

Remote smoke:

```bash
rsync -av experiments/gemma4_IO_behaviors/mem8G/ nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_IO_behaviors/mem8G/
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && RUN_TIMESTAMP=pipe_smoke N_PREDICT=4 PROMPT_LIMIT=1 THREAD_SWEEP=8,32 THREAD_RUNS=1 WARMUP_RUNS=0 RUNTIME_REPEATS=1 BASE_PORT=8110 experiments/gemma4_IO_behaviors/mem8G/run_mem8g_pipeline_calibration.sh'
```

Remote full calibration:

```bash
rsync -av experiments/gemma4_IO_behaviors/mem8G/ nthu-cs:~/workspace/HuangWayne/project/experiments/gemma4_IO_behaviors/mem8G/
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && RUN_TIMESTAMP=pipe_n64 N_PREDICT=64 PROMPT_LIMIT=1 THREAD_SWEEP=8,16,24,32 THREAD_RUNS=3 WARMUP_RUNS=1 RUNTIME_REPEATS=3 BASE_PORT=8110 experiments/gemma4_IO_behaviors/mem8G/run_mem8g_pipeline_calibration.sh'
```

For a longer decode window, set `N_PREDICT=128`. The top-level
`mem8G/REPORT.md` is replaced with the latest calibration report after
aggregation.

Raw prompt outputs are written under:

```text
experiments/gemma4_IO_behaviors/mem8G/statistics/decode_expert_cache_fault_<timestamp>/
```

Matrix SVGs are written only under:

```text
experiments/gemma4_IO_behaviors/mem8G/figures/matrix_<timestamp>/
```

For a single prompt, the growth renderer writes
`expert_cache_matrices_growth.svg`. For multiple prompts, it writes one
`prompt_<index>_<name>_growth.svg` per prompt.

Each prompt statistics directory includes:

- `expert_cache_matrices.json`
- `expert_cache_matrix_samples.csv`
- `decode_moe_vs_non_moe_refaults.csv`
- `decode_refaults_by_tensor.csv`
- `run_meta.json`

The growth renderer uses blue for already resident experts, red for newly
resident experts, yellow for experts that were resident in the previous sample
but nonresident in the current sample, and light gray for nonresident experts.

## Local Checks

```bash
python3 -m py_compile experiments/gemma4_IO_behaviors/mem8G/utils/*.py
bash -n experiments/gemma4_IO_behaviors/mem8G/run_mem8g_decode_expert_cache_fault.sh
bash -n experiments/gemma4_IO_behaviors/mem8G/run_mem8g_decode_resident_timeline.sh
bash -n experiments/gemma4_IO_behaviors/mem8G/run_mem8g_rpp_admission_sim.sh
bash -n experiments/gemma4_IO_behaviors/mem8G/run_mem8g_runtime_rpp_prefetch.sh
bash -n experiments/gemma4_IO_behaviors/mem8G/run_mem8g_runtime_rpp_prefetch_matrix.sh
bash -n experiments/gemma4_IO_behaviors/mem8G/run_mem8g_pipeline_calibration.sh
python3 experiments/gemma4_IO_behaviors/mem8G/utils/simulate_rpp_expert_admission.py --help
python3 experiments/gemma4_IO_behaviors/mem8G/utils/runtime_rpp_prefetch_probe.py --help
```
