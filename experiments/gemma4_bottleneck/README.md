# Gemma4 26B Prefill/Decode Bottleneck Benchmark

This benchmark attaches to a running `llama-server` and measures inference in
two phases:

- `prefill`: request start until llama-server reports prompt processing done.
- `decode`: prompt processing done until the streaming completion finishes.

If the server does not emit prompt progress events, the script falls back to
the first streamed token as the boundary. That fallback includes the first
decode step in the prefill bucket, so prefer a recent `llama-server` with
`return_progress` support.

## Start llama-server

```bash
scripts/llama-test.sh server \
  -m models/gemma4-26B.gguf \
  -c 8192 \
  -ngl 0 \
  --port 8080 \
  -- --no-warmup
```

In another terminal, find the server PID:

```bash
pgrep -af llama-server
```

## Run The Benchmark

```bash
python3 experiments/gemma4_bottleneck/benchmark_prefill_decode.py \
  --pid <LLAMA_SERVER_PID> \
  --url http://127.0.0.1:8080/completion \
  --runs 5 \
  --warmup-runs 1 \
  --n-predict 128
```

Outputs:

- `experiments/gemma4_bottleneck/prefill_decode_benchmark.jsonl`
- `experiments/gemma4_bottleneck/prefill_decode_benchmark.csv`

Use a fixed prompt file when comparing settings:

```bash
python3 experiments/gemma4_bottleneck/benchmark_prefill_decode.py \
  --pid <LLAMA_SERVER_PID> \
  --prompt-file prompts/long_context.txt \
  --runs 5 \
  --n-predict 256
```

## Reading The Signals

Important columns:

- `wall_s`: elapsed time for the phase.
- `total_cpu_s`: CPU time summed across llama-server threads.
- `cpu_parallelism`: `total_cpu_s / wall_s`; can exceed 1.0 with many CPU threads.
- `block_io_delay_s`: kernel block I/O delay if task delay accounting is enabled.
- `major_faults`: mmap demand paging from storage; high values usually mean model
  pages or KV memory are being faulted in.
- `read_mb`: bytes read from storage according to `/proc/<pid>/io`.
- `bound_guess`: simple heuristic from CPU activity and I/O counters.

Typical interpretation:

- High `major_faults`, `read_mb`, or `block_io_delay_s` during `prefill` means
  cold mmap/page-cache I/O is involved.
- Warm repeated runs with low I/O counters and high `cpu_parallelism` are CPU
  compute bound.
- High `system_cpu_s` can point to kernel overhead, memory mapping, page faults,
  or thread scheduling overhead.
- If all accounting is low but wall time is high, inspect server logs and whether
  the process is blocked outside kernel I/O accounting.

For I/O diagnosis, compare the first measured run after starting the server with
later warm runs. Do not enable `--cache-prompt` unless you specifically want to
measure prompt-cache reuse; it changes the prefill workload.

## Tensor Residency Monitor

To identify which mapped GGUF tensors churn under low RAM, enable the pagemap
monitor in the container case runner:

```bash
ENABLE_TENSOR_SWAP_MONITOR=1 \
TENSOR_MONITOR_INTERVAL=1.0 \
CASE_NAME=ram8g_swap8g \
OUT_DIR=experiments/gemma4_bottleneck/results/example/ram8g_swap8g \
experiments/gemma4_bottleneck/run_container_ram_case.sh
```

Extra outputs are written under `$OUT_DIR/tensor_residency/`:

- `tensor_residency_totals.csv`: total present/nonresident/swapped MB per sample.
- `tensor_residency_samples.csv`: per-tensor state and transition deltas.
- `tensor_residency_summary.csv`: tensors sorted by total evict/refault churn.
- `family_layer_residency_summary.csv`: same signal grouped by family/layer/type.

For model mmap weights, pages evicted from RAM usually appear as
`nonresident`, not `swapped`, because clean file-backed pages are dropped and
later faulted back from the GGUF file. Use `total_evicted_mb` and
`total_refaulted_mb` to find tensors that repeatedly leave RAM and return during
the 10G/8G/6G runs. `total_swapout_mb` is still reported for cases where the
kernel marks pages swapped.

After running multiple RAM limits, combine them with:

```bash
python3 experiments/gemma4_bottleneck/compare_tensor_residency.py \
  experiments/gemma4_bottleneck/results/<sweep>/ram10g_swap8g \
  experiments/gemma4_bottleneck/results/<sweep>/ram8g_swap8g \
  experiments/gemma4_bottleneck/results/<sweep>/ram6g_swap8g \
  --output-dir experiments/gemma4_bottleneck/results/<sweep>/tensor_residency_compare
```

The main cross-case files are `tensor_residency_across_cases.csv` and
`family_layer_residency_across_cases.csv`, sorted by total evict/refault churn.
