# mem8G RPP Expert Admission Simulation

Offline 8G simulation using true router labels and global RPP predictions. It estimates demand-path expert loads; it does not measure wall-clock llama.cpp latency.

## Config

```json
{
  "batch_size": 24,
  "cache_capacity_override": 0,
  "capacity_config": "experiments/gemma4_IO_behaviors/mem8G/expert_capacity/statistics/capacity_estimate_budget_model_decode500_recal_20260519_1150/run_config.json",
  "checkpoint": "experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt",
  "config": "experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json",
  "data_root": "dataset/prompt10000/router_label_npz/npz",
  "data_summary": {
    "files": 1000,
    "loss_tokens": 9727,
    "max_token_id": 255968,
    "seq_len_gt_512": 0,
    "source_counts": {
      "tatsu-lab/alpaca": 1000
    },
    "task_counts": {
      "instruction": 1000
    },
    "tokens": 38757
  },
  "device_requested": "auto",
  "device_resolved": "cpu",
  "expert_bytes": 0,
  "num_workers": 0,
  "predict_topk": 8,
  "prefetch_budgets": [
    60,
    120,
    180
  ],
  "prefetch_thresholds": [],
  "samples": 1000,
  "torch_num_threads": 8,
  "wall_s": 12.162845373153687
}
```

## Summary

| strategy | tokens | demand_loads | prefetch_loads | total_loads | demand_miss_rate | prefetch_candidate_rate | demand_load_reduction_vs_lru | total_load_reduction_vs_lru | capacity_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| demand_lru | 9727 | 425589 | 0 | 425589 | 0.182306 | 0.000000 | 0.000000 | 0.000000 | 1184.0 |
| rpp_prefetch_topk | 9727 | 206469 | 335446 | 541915 | 0.088443 | 1.000000 | 0.514863 | -0.273329 | 1184.0 |
| rpp_prefetch_budget_60 | 9727 | 384717 | 42795 | 427512 | 0.164798 | 0.250000 | 0.096036 | -0.004518 | 1184.0 |
| rpp_prefetch_budget_120 | 9727 | 324258 | 115214 | 439472 | 0.138899 | 0.500000 | 0.238096 | -0.032621 | 1184.0 |
| rpp_prefetch_budget_180 | 9727 | 260145 | 207735 | 467880 | 0.111436 | 0.750000 | 0.388741 | -0.099371 | 1184.0 |
| oracle_prefetch | 9727 | 0 | 425589 | 425589 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1184.0 |
| oracle_logits_prefetch | 9727 | 558 | 427631 | 428189 | 0.000239 | 1.000000 | 0.998689 | -0.006109 | 1184.0 |

## Interpretation

- Best demand-path reduction: `rpp_prefetch_topk` removes 51.486% of LRU demand loads.
- Best total-load result: `rpp_prefetch_budget_60` changes total expert loads by -0.452%; negative means prefetch waste exceeds saved demand loads.
- Runtime prefetch should start from a strategy with positive demand-load reduction and acceptable total-load overhead, then measure elapsed decode time in Docker 8G.

## Truth Prefill Logits 5 Short N16 (2026-06-23)

- Path: `experiments/gemma4_IO_behaviors/mem8G/statistics/truth_prefill_logits_5_short_n16_20260623_040619`
- Prompt ids: `alpaca_train_000000, alpaca_train_000001, alpaca_train_000002, alpaca_train_000004, alpaca_train_000006`
- Record indices: `0, 1, 2, 4, 6`
- Selection note: literal `len(prompt_text) <= 50` had `0` matches; used approximately short instruction payloads, preserving original prompt rows.
- Prompt chars: payload `35, 45, 49, 46, 49`; full `prompt_text` `67, 77, 98, 78, 153`
- Generation: `5/5 ok`, `n_predict=16`, generated chars `68, 40, 54, 48, 65`
- Truth prefill labels: `5 / 5` NPZ ok, tokens `199`, loss tokens `78`
- Validation: `ok`; checked required arrays, logits/topk shapes, full 30-layer mask, positive loss masks, and `generation_settings.n_predict == 16`.
- Reports: `experiments/gemma4_IO_behaviors/mem8G/statistics/truth_prefill_logits_5_short_n16_20260623_040619/REPORT.md`, `experiments/gemma4_IO_behaviors/mem8G/statistics/truth_prefill_logits_5_short_n16_20260623_040619/npz_validation_summary.json`

## Mem6G Prefill-Only mmap Density Sweep (2026-06-23)

- Report: `experiments/gemma4_IO_behaviors/mem8G/statistics/mem6g_prefill_mmap_density_report_20260623.md`
- Setup: `--memory=6g --memory-swap=6g`, `PROMPT_LIMIT=5`, `POOL_SIZE=5`, `N_PREDICT=16`, `RPP_PREFETCH_PREFILL_ONLY=1`, `RPP_PREFETCH_IO_MODE=mmap`.
- Compared density `0.15`, `0.3`, `0.45`, `0.6` against one shared `baseline_fifo` run.
- Validation for density runs: global order mismatch `0`, ubatch mismatch `0`, prefill slot mismatch `0`.
- Best result: density `0.15` reduced cgroup pgmajfault by `7,784` (`0.75%`) and total wall time by `1.47s` (`1.11%`) vs baseline.
- Higher densities regressed: density `0.45` and `0.6` increased both page faults and latency.

## Prefill+Decode RPP mmap Touch Smoke (2026-06-24)

- Path: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_scheduler_compare_20260624_113216`
- Setup: `PROMPT_LIMIT=1`, `N_PREDICT=8`, `PREDICT_TOPK=8`, `RPP_PREFETCH_IO_MODE=mmap`, `RPP_PREFETCH_PREFILL_ONLY=0`, `RPP_PREFETCH_COUNT_THRESHOLD=1`, `RPP_UBATCH_PREFETCH_MIN_COUNT=1`, `RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD=0.0`, `HOST_DROP_CACHES=0`.
- Build flow: host-side remote CMake/make in `llama.cpp/build-rpp-live-host`; Docker is used only for runtime inference.
- Oracle trace: prefill-only true labels, `rows=17`, `samples=1`, `include_decode=false`.
- Baseline `baseline_fifo_k5`: wall `26.025960s`, throughput `0.307385 tok/s`, cgroup pgmajfault delta `118,268`, cgroup pgfault delta `501,454`.
- Decode RPP mmap `decode_rpp_top8_mmap_ubatch_k5`: wall `33.604002s`, throughput `0.238067 tok/s`, cgroup pgmajfault delta `115,299`, cgroup pgfault delta `789,815`.
- RPP/mmap validation: `prefetch_io_mode=mmap`, `prefetch_prefill_only=false`, `decode_calls=9`, `decode_rpp_prediction_rows=7`, `decode_rpp_missing_prediction_rows=0`, `ubatch_prefetch_events=3,466`, `ubatch_counter_rows=3,466`, `advised_bytes=13,671,815,168`, `fadvise_errors=0`.
- Prediction alignment counters: `oracle_mismatches=0`, `global_order_prediction_mismatches=0`, `ubatch_prediction_mismatches=0`, `decode_token_slot_mismatches=0`.
- Interpretation: the end-to-end prefill+decode mmap touch path is functional, and decode tokens are supplied by live RPP top8 rather than decode oracle lookup. This smoke size is too small for latency conclusions; it only validates the runtime path.

## Prefill+Decode RPP mmap Density 0.15 Comparison (2026-06-24)

- Path: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_scheduler_compare_20260624_113855`
- Code change: ubatch density-derived required count now clamps to at least `1`: `max(1, ceil(max(1, token_count) * RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD))`, then combines with `RPP_UBATCH_PREFETCH_MIN_COUNT`.
- Setup: `PROMPT_LIMIT=5`, `N_PREDICT=16`, `PREDICT_TOPK=8`, `RPP_PREFETCH_IO_MODE=mmap`, `RPP_PREFETCH_PREFILL_ONLY=0`, `RPP_PREFETCH_COUNT_THRESHOLD=1`, `RPP_UBATCH_PREFETCH_MIN_COUNT=1`, `RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD=0.15`, `HOST_DROP_CACHES=0`.
- Oracle trace: prefill-only true labels, `rows=121`, `samples=5`, `include_decode=false`.
- Baseline `baseline_fifo_k5`: wall `106.407715s`, throughput `0.751825 tok/s`, p50 `103.448843s`, p95 `105.814532s`, cgroup pgfault delta `3,750,486`, cgroup pgmajfault delta `880,469`.
- Decode RPP mmap `decode_rpp_top8_mmap_ubatch_k5`: wall `137.240278s`, throughput `0.582919 tok/s`, p50 `132.998930s`, p95 `137.237799s`, cgroup pgfault delta `4,569,626`, cgroup pgmajfault delta `836,669`.
- Delta vs baseline: wall `+30.832563s` (`+28.98%`), throughput `-22.47%`, cgroup pgmajfault `-43,800` (`-4.97%`), cgroup pgfault `+819,140` (`+21.84%`).
- RPP/mmap validation: `prefetch_io_mode=mmap`, `prefetch_prefill_only=false`, `ubatch_prefetch_density_threshold=0.15`, `decode_calls=18`, `decode_rpp_prediction_rows=75`, `decode_rpp_missing_prediction_rows=0`, `ubatch_prefetch_events=12,508`, `ubatch_counter_rows=14,355`, `advised_bytes=49,335,351,296`, `fadvise_errors=0`.
- Density trace check: `ubatch_prefetch_events.csv` used `density_threshold=0.150000`; observed required counts included `10`, `2`, and `1`, matching the density-derived clamp for different ubatch token counts.
- Prediction alignment counters: `oracle_mismatches=0`, `global_order_prediction_mismatches=11`, `ubatch_prediction_mismatches=0`, `decode_token_slot_mismatches=0`.
- Interpretation: density `0.15` reduces major faults versus baseline in this 5-prompt run, but the current synchronous mmap touch path adds enough overhead that wall time regresses.

## Async Prefill+Decode RPP mmap Density 0.15 (2026-06-24)

- Path: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_scheduler_compare_20260624_115338`
- Code change: added optional async prefetch worker behind `RPP_PREFETCH_ASYNC=1`. Decode path now enqueues mmap touch jobs into a bounded queue and returns; worker records actual completion in `ubatch_prefetch_async_trace.csv`.
- Setup: `PROMPT_LIMIT=5`, `N_PREDICT=16`, `PREDICT_TOPK=8`, `RPP_PREFETCH_IO_MODE=mmap`, `RPP_PREFETCH_PREFILL_ONLY=0`, `RPP_PREFETCH_COUNT_THRESHOLD=1`, `RPP_UBATCH_PREFETCH_MIN_COUNT=1`, `RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD=0.15`, `RPP_PREFETCH_ASYNC=1`, `RPP_PREFETCH_QUEUE_CAP=512`, `RPP_UBATCH_PREFETCH_MAX_EXPERTS=0`, `RUN_BASELINE=0`, `RPP_POST_CLIENT_SLEEP_S=5`, `HOST_DROP_CACHES=0`.
- Async mmap `decode_rpp_top8_mmap_async_ubatch_k5`: wall `95.491877s`, throughput `0.837768 tok/s`, p50 `95.489940s`, p95 `95.491327s`, cgroup pgfault delta `3,774,013`, cgroup pgmajfault delta `726,169`.
- Compared with previous same-size baseline `baseline_fifo_k5` from `live_rpp_scheduler_compare_20260624_113855`: wall `-10.915838s` (`-10.26%`), throughput `+11.43%`, cgroup pgmajfault `-154,300` (`-17.52%`), cgroup pgfault `+23,527` (`+0.63%`).
- Compared with previous sync mmap density `0.15`: wall `-41.748401s` (`-30.42%`), throughput `+43.72%`, cgroup pgmajfault `-110,500` (`-13.21%`), cgroup pgfault `-795,613` (`-17.41%`).
- RPP/mmap validation: `prefetch_async=true`, `prefetch_async_queue_cap=512`, `ubatch_prefetch_density_threshold=0.15`, `decode_calls=17`, `decode_rpp_prediction_rows=75`, `decode_rpp_missing_prediction_rows=0`, `ubatch_prefetch_events=12,159`, `advised_bytes=33,507,110,912`, `fadvise_errors=0`.
- Async queue validation: `async_prefetch_enqueued_rows=8,512`, `async_prefetch_completed_rows=8,512`, `async_prefetch_dropped_rows=3,647`; async trace status counts were `completed=8,512`, `dropped_queue_full=3,647`.
- Prediction alignment counters: `oracle_mismatches=0`, `global_order_prediction_mismatches=0`, `ubatch_prediction_mismatches=0`, `decode_token_slot_mismatches=0`.
- Interpretation: moving mmap touch off the decode critical path fixes the sync-path regression for this run. Queue bounding also reduces actual touched bytes versus sync mmap; the remaining tuning question is queue cap / max experts tradeoff, since 3,647 planned jobs were dropped.

## Priority Prefetch Worklist + Reclaim Trace Smoke (2026-06-24)

- Path: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_priority_smoke_ub4_codex_20260624`
- Code change: async mmap prefetch queue now uses a bounded priority worklist; pre-decode planning splits the current `batch_view` into physical ubatches, and reclaim emits trace-only priority candidates via `RPP_RECLAIM_MODE=trace-priority`.
- Setup: `PROMPT_LIMIT=1`, `N_PREDICT=8`, `RUN_BASELINE=0`, `HOST_DROP_CACHES=0`, `RPP_PREFETCH_ASYNC=1`, `RPP_PREFETCH_QUEUE_POLICY=priority`, `RPP_RECLAIM_MODE=trace-priority`, `RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD=0.15`, `RPP_UBATCH_PREFETCH_MIN_COUNT=1`, `LLAMA_EXTRA_ARGS='-ub 4'`.
- Remote build: host-side `LLAMA_RPP_LIVE=ON` CMake build passed on `nthu-cs`; Docker/Podman was used for runtime inference only.
- Runtime result: `decode_rpp_top8_mmap_async_ubatch_k5` wall `32.476004s`, throughput `0.246336 tok/s`, cgroup pgfault delta `426,958`, cgroup pgmajfault delta `92,139`.
- Plan validation: `ubatch_plan_rows=12`, observed target ubatch ids `[0, 1, 2]`, confirming multi-ubatch planning from one decode batch view.
- Priority queue validation: `prefetch_priority_planned_rows=834`, `prefetch_priority_enqueued_rows=746`, `prefetch_priority_completed_rows=513`, `prefetch_priority_dropped_low_priority_rows=88`, `prefetch_priority_replaced_rows=233`, `prefetch_priority_stale_rows=0`, `advised_bytes=2,024,715,264`.
- Reclaim validation: `reclaim_mode=trace-priority`, `reclaim_lookahead_ubatches=4`, `reclaim_priority_candidates=4,608`, `reclaim_priority_trace_rows=5,120`; no real eviction was performed.
- RPP/mmap validation: `decode_calls=10`, `decode_rpp_prediction_rows=7`, `decode_rpp_missing_prediction_rows=0`, `fadvise_errors=0`.
- Interpretation: priority replacement and trace-only reclaim are functional in a forced multi-ubatch smoke. This run is only a correctness smoke; latency is not comparable to previous default-ubatch runs because `-ub 4` intentionally changes physical batching.

## Priority Reclaim DONTNEED Thrashing Probe (2026-06-24)

- Paths:
  - default ubatch trace-only: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_reclaim_trace_cmp_codex_20260624`
  - default ubatch DONTNEED: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_reclaim_dontneed_cmp_codex_20260624`
  - `-ub 4` trace-only: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_reclaim_trace_ub4_cmp_codex_20260624`
  - `-ub 4` DONTNEED: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_reclaim_dontneed_ub4_cmp_codex_20260624`
- Code change: added non-default `RPP_RECLAIM_MODE=fadvise-dontneed`, which runs `POSIX_FADV_DONTNEED` only for reclaim candidates that pass inflight/protected/future-high-priority filters. Default remains `trace-priority`.
- Common setup: `PROMPT_LIMIT=5`, `N_PREDICT=16`, `RUN_BASELINE=0`, `HOST_DROP_CACHES=0`, `RPP_PREFETCH_ASYNC=1`, `RPP_PREFETCH_QUEUE_POLICY=priority`, `RPP_RECLAIM_QUEUE_CAP=64`, `RPP_RECLAIM_LOOKAHEAD_UBATCHES=4`, `RPP_RECLAIM_PROTECT_UBATCHES=2`, `RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD=0.15`, `RPP_UBATCH_PREFETCH_MIN_COUNT=1`.
- Default ubatch comparison: DONTNEED reclaimed `4,718,264,320` bytes with `2,176` fadvise calls and `0` reclaim errors. Wall time changed from `112.707795s` to `113.715347s` (`+0.89%`), cgroup pgmajfault changed from `864,807` to `869,356` (`+0.53%`), and cgroup pgfault changed from `3,774,304` to `3,791,704` (`+0.46%`). This path had no priority prefetch completed rows, so it mostly measures standalone reclaim overhead.
- `-ub 4` prefetch+reclaim comparison: DONTNEED reclaimed `4,995,809,280` bytes with `2,304` fadvise calls and `0` reclaim errors. Wall time changed from `160.735544s` to `161.413443s` (`+0.42%`), while cgroup pgmajfault changed from `806,474` to `765,944` (`-5.03%`) and cgroup pgfault changed from `3,618,249` to `3,469,207` (`-4.12%`).
- `-ub 4` prefetch validation: both runs completed `4,899` priority prefetch jobs, had `0` stale rows, `0` decode RPP missing predictions, and `0` prefetch fadvise errors. DONTNEED advised bytes for prefetch were `19,272,895,488`, close to trace-only prefetch advised bytes `19,262,487,552`.
- Interpretation: with `RPP_RECLAIM_QUEUE_CAP=64`, real DONTNEED did not show a clear thrashing signature in the forced multi-ubatch run; faults decreased while wall time regressed slightly, likely from extra fadvise/reclaim overhead. This is a single-run probe, so repeat runs and a cap sweep are needed before enabling real reclaim by default.

## FIFO vs Priority Prefetch Queue A/B (2026-06-24)

- Paths:
  - FIFO: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_queue_fifo_ub4_cmp_codex_20260624`
  - priority: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_queue_priority_ub4_cmp_codex_20260624`
- Code change: `RPP_PREFETCH_QUEUE_POLICY=fifo` now uses FIFO pop and drops incoming work when the bounded queue is full; `priority` keeps highest-priority pop plus replacement of the current lowest-priority queued job. No real reclaim was used.
- Common setup: `PROMPT_LIMIT=5`, `N_PREDICT=16`, `RUN_BASELINE=0`, `HOST_DROP_CACHES=0`, `RPP_PREFETCH_ASYNC=1`, `RPP_PREFETCH_QUEUE_CAP=512`, `RPP_RECLAIM_MODE=trace-priority`, `RPP_RECLAIM_QUEUE_CAP=64`, `RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD=0.15`, `RPP_UBATCH_PREFETCH_MIN_COUNT=1`, `LLAMA_EXTRA_ARGS='-ub 4'`.
- Runtime result: FIFO wall `154.637177s`, throughput `0.517340 tok/s`, cgroup pgfault `3,531,166`, cgroup pgmajfault `777,948`. Priority wall `153.765668s`, throughput `0.520272 tok/s`, cgroup pgfault `3,422,864`, cgroup pgmajfault `771,482`.
- Delta priority vs FIFO: wall `-0.871509s` (`-0.56%`), throughput `+0.57%`, cgroup pgfault `-108,302` (`-3.07%`), cgroup pgmajfault `-6,466` (`-0.83%`).
- Queue behavior: both completed `4,899` mmap touch jobs with `0` stale rows and `0` fadvise errors. FIFO planned `11,711`, enqueued `4,899`, dropped queue-full `6,812`, replaced `0`. Priority planned `13,348`, enqueued `6,455`, dropped low-priority `6,893`, replaced `1,556`.
- Completed job quality: priority completed higher-value work than FIFO. Completed density mean was `0.930326` for priority vs `0.818075` for FIFO; completed counter mean was `1.595019` for priority vs `1.214738` for FIFO. Dropped density mean was lower for priority (`0.354638`) than FIFO (`0.417976`).
- Interpretation: in this single controlled `-ub 4` run, priority queue behaved as intended and was modestly better than FIFO on wall time and page faults. The gain is small, so repeat runs or larger prompt sets are needed to separate signal from run-to-run variance.

## Baseline vs Async FIFO vs Async Priority, No Real Evict (2026-06-24)

- Paths:
  - default ubatch baseline/FIFO: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_baseline_fifo_default_cmp_codex_20260624`
  - default ubatch priority: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_priority_default_cmp_codex_20260624`
  - `-ub 32` baseline/FIFO: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_baseline_fifo_ub32_cmp_codex_20260624`
  - `-ub 32` priority: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_priority_ub32_cmp_codex_20260624`
- Common setup: `PROMPT_LIMIT=5`, `N_PREDICT=16`, `HOST_DROP_CACHES=0`, `RPP_PREFETCH_ASYNC=1`, `RPP_PREFETCH_QUEUE_CAP=512`, `RPP_RECLAIM_MODE=trace-priority`, `RPP_RECLAIM_QUEUE_CAP=64`, `RPP_RECLAIM_LOOKAHEAD_UBATCHES=4`, `RPP_RECLAIM_PROTECT_UBATCHES=2`, `RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD=0.15`, `RPP_UBATCH_PREFETCH_MIN_COUNT=1`. No `RPP_RECLAIM_MODE=fadvise-dontneed` was used.
- Default ubatch result: baseline wall `117.277590s`, FIFO async wall `115.392339s`, priority async wall `117.129466s`. FIFO improved wall time by `1.61%` vs baseline; priority improved wall time by only `0.13%` vs baseline and was `1.51%` slower than FIFO.
- Default ubatch page faults: baseline cgroup pgfault `3,857,593`, pgmajfault `906,877`; FIFO pgfault `3,826,509`, pgmajfault `892,929`; priority pgfault `3,838,463`, pgmajfault `881,905`.
- Default ubatch planner validation: both FIFO and priority had `ubatch_plan_rows=17`, but observed only `target_ubatch_id=[0]`. Both had `prefetch_priority_enqueued_rows=0`, `prefetch_priority_completed_rows=0`, `prefetch_priority_dropped_low_priority_rows=0`, `prefetch_priority_replaced_rows=0`, and `prefetch_priority_stale_rows=0`.
- Interpretation for default ubatch: current v1 priority planner only schedules future physical ubatches inside the current `batch_view`. With default batching in this run, each submitted plan had only target ubatch `0`, so there was no future ubatch passing the min-lead gate. The priority queue therefore had no prefetch jobs to sort; the small FIFO improvement is from the remaining async path/runtime variance, not from priority scheduling.
- `-ub 32` result: baseline wall `118.687594s`, FIFO async wall `118.421844s`, priority async wall `119.869592s`. FIFO improved wall time by `0.22%` vs baseline; priority regressed by `0.996%` vs baseline and was `1.22%` slower than FIFO.
- `-ub 32` queue validation: FIFO had `ubatch_plan_rows=20`, observed target ubatch ids `[0, 1, 2, 3]`, `prefetch_priority_enqueued_rows=513`, `prefetch_priority_completed_rows=513`, and `prefetch_priority_dropped_low_priority_rows=1122`. Priority had `ubatch_plan_rows=18`, observed target ubatch ids `[0, 1]`, `prefetch_priority_enqueued_rows=499`, `prefetch_priority_completed_rows=499`, and no low-priority drops or replacements.
- `-ub 32` page faults: FIFO pgfault `3,676,981`, pgmajfault `865,851`; priority pgfault `3,784,341`, pgmajfault `875,631`. Priority was worse than FIFO on both wall time and faults in this single run.
- Overall conclusion: priority queue implementation is async and functional, but current v1 planning does not produce useful work under default ubatch. Forced smaller ubatch settings create future physical ubatches, but that changes the execution shape and is not a clean performance win. The next implementation step should be cross-decode-call planning so default ubatch can enqueue genuinely future expert work without relying on artificial `-ub` limits.

## Current-Ubatch Async Prefetch Restore Smoke (2026-06-24)

- Path: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_current_ubatch_fifo_smoke_codex_20260624`
- Code change: changed default `RPP_PREFETCH_MIN_LEAD_UBATCHES` from `1` to `0` in both `server-rpp-live.cpp` and `run_mem8g_live_rpp_scheduler.sh`. This allows the current physical ubatch to enqueue async mmap touch work. Future ubatch work is still allowed, but current ubatch jobs have `target_ubatch_distance=0` and therefore sort before farther future ubatches in priority mode.
- Setup: `PROMPT_LIMIT=1`, `N_PREDICT=8`, `RUN_BASELINE=0`, `HOST_DROP_CACHES=0`, default ubatch, `RPP_PREFETCH_ASYNC=1`, `RPP_PREFETCH_QUEUE_POLICY=fifo`, `RPP_PREFETCH_MIN_LEAD_UBATCHES=0`, `RPP_RECLAIM_MODE=trace-priority`, `RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD=0.15`, `RPP_UBATCH_PREFETCH_MIN_COUNT=1`.
- Build: host-side remote `llama-server` build passed after the change; runtime remained inside Docker/Podman.
- Runtime result: wall `23.398437s`, cgroup pgfault `610,758`, cgroup pgmajfault `90,646`.
- Planner validation: `ubatch_plan_rows=9`, observed target ubatch ids `[0]`, confirming default batching still has only the current ubatch in this smoke.
- Async queue validation: `ubatch_prefetch_events=2,896`, `async_prefetch_enqueued_rows=2,706`, `async_prefetch_completed_rows=2,706`, `async_prefetch_dropped_rows=190`, `advised_bytes=10,642,114,560`, `decode_rpp_missing_prediction_rows=0`, `fadvise_errors=0`.
- Interpretation: the earlier no-op behavior was caused by the min-lead gate excluding current ubatch work. With lead `0`, default ubatch once again generates async mmap work, making the new FIFO/priority queue policies comparable to the original async queue path.

## Current-Ubatch-Only Prefetch Cleanup Smoke (2026-06-24)

- Path: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_current_only_priority_smoke_codex_20260624`
- Code cleanup: removed future-ubatch prefetch gating from the public prefetch path. The prefetch queue now only receives jobs for the current physical ubatch (`target_ubatch_id == source_ubatch_id`). Removed the temporary `RPP_PREFETCH_CURRENT_UBATCH_ONLY` / `RPP_PREFETCH_MIN_LEAD_UBATCHES` script/env plumbing and deleted the unused duplicate current-ubatch prefetch function. Reclaim trace remains separate from the prefetch queue.
- Setup: `PROMPT_LIMIT=1`, `N_PREDICT=8`, `RUN_BASELINE=0`, default ubatch, `RPP_PREFETCH_ASYNC=1`, `RPP_PREFETCH_QUEUE_POLICY=priority`, `RPP_RECLAIM_MODE=trace-priority`, `RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD=0.15`, `RPP_UBATCH_PREFETCH_MIN_COUNT=1`.
- Build: local host-side `llama-server` build passed; remote host-side build on `nthu-cs` also passed before Docker/Podman runtime.
- Runtime result: wall `24.293421s`, cgroup pgfault `626,369`, cgroup pgmajfault `95,020`.
- Queue validation: `ubatch_plan_rows=9`, `ubatch_prefetch_events=2,896`, `prefetch_priority_enqueued_rows=2,706`, `prefetch_priority_completed_rows=2,706`, `prefetch_priority_dropped_low_priority_rows=190`, `prefetch_priority_replaced_rows=0`, `advised_bytes=10,642,114,560`.
- Current-only validation: `prefetch_priority_trace.csv` active rows (`planned`, `enqueued`, `completed`) all had `target_ubatch_distance=0`; `non_current_active_rows=0`. This confirms no future ubatch work entered the prefetch queue.
- RPP validation: `decode_rpp_missing_prediction_rows=0`, `fadvise_errors=0`.

## Current-Ubatch-Only Baseline vs FIFO vs Priority (2026-06-24)

- Paths:
  - baseline + FIFO: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_baseline_fifo_current_only_cmp_codex_20260624`
  - priority: `experiments/gemma4_IO_behaviors/mem8G/statistics/live_rpp_priority_current_only_cmp_codex_20260624`
- Common setup: `PROMPT_LIMIT=5`, `N_PREDICT=16`, `BASELINE_DISPATCH_MODE=rolling`, `LIVE_DISPATCH_MODE=rolling`, `POOL_SIZE=24`, `HOST_DROP_CACHES=0`, `RPP_PREFETCH_ASYNC=1`, `RPP_PREFETCH_QUEUE_CAP=512`, `RPP_PREFETCH_MAX_STALENESS_UBATCHES=0`, `RPP_RECLAIM_MODE=trace-priority`, `RPP_RECLAIM_QUEUE_CAP=64`, `RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD=0.15`, `RPP_UBATCH_PREFETCH_MIN_COUNT=1`. No real reclaim eviction was used.
- Wall time: baseline `119.662051s`, async FIFO `93.382713s`, async priority `99.474411s`.
- Delta vs baseline: FIFO improved wall time by `26.279338s` (`21.96%`), cgroup pgmajfault by `179,358` (`19.63%`), and cgroup pgfault by `98,601` (`2.56%`). Priority improved wall time by `20.187639s` (`16.87%`), cgroup pgmajfault by `198,532` (`21.73%`), and cgroup pgfault by `143,205` (`3.72%`).
- Priority vs FIFO: priority was `6.091699s` slower (`6.52%`) than FIFO, despite lower cgroup pgmajfault by `19,174` (`2.61%`) and lower cgroup pgfault by `44,604` (`1.19%`).
- FIFO queue counters: `ubatch_prefetch_events=12,203`, `prefetch_priority_enqueued_rows=8,512`, `prefetch_priority_completed_rows=8,512`, `prefetch_priority_dropped_low_priority_rows=3,691`, `prefetch_priority_replaced_rows=0`, `prefetch_priority_stale_rows=0`, `advised_bytes=33,504,880,640`.
- Priority queue counters: `ubatch_prefetch_events=12,159`, `prefetch_priority_enqueued_rows=8,512`, `prefetch_priority_completed_rows=8,512`, `prefetch_priority_dropped_low_priority_rows=3,647`, `prefetch_priority_replaced_rows=0`, `prefetch_priority_stale_rows=0`, `advised_bytes=33,507,110,912`.
- Current-only validation: both FIFO and priority had `non_current_active_rows=0`; active prefetch rows only used current ubatch work.
- Completed job quality: FIFO completed density mean `0.375231`, counter mean `3.698308`; priority completed density mean `0.376265`, counter mean `3.703477`. Dropped density mean was essentially identical: FIFO `0.199580`, priority `0.199575`.
- Interpretation: current-ubatch async prefetch is working again and clearly beats baseline in this run. Priority did not improve wall time over FIFO because both policies completed the same number of jobs with nearly identical completed-job quality; priority reduced page faults slightly, but not enough to offset runtime effects. The next tuning target is not future ubatch planning, but why priority's better fault profile does not translate into faster wall time under the current queue cap and current-ubatch-only workload.

## Short 1000 Truth Prompt Folder (2026-06-24)

- Path: `experiments/gemma4_IO_behaviors/mem8G/statistics/truth_prefill_logits_prompt1000_short_plus_5_20260624`
- Layout: `short_1000/` contains a new 1000-sample short-prompt truth subset; `short_5/` is a symlink to the original 5-short-prompt truth root. They are under the same parent folder but not mixed into one manifest.
- Source: selected from existing remote `dataset/prompt10000` router-label NPZ files; no new logits dump was run. NPZ files in `short_1000/router_label_npz/npz` are hardlinks to the existing prompt10000 NPZ files.
- Selection: preserved the prompt10000 source mix at `300/200/200/200/100` for alpaca/xsum/wmt/code/math, choosing the shortest prompt-token samples per source.
- Result: `short_1000` has `1000` prompts, `1000` NPZ files, and a prefill-only `oracle_live_trace.jsonl` with `39,329` rows.
- Lengths: selected prompt-token min/max/mean is `22 / 153 / 49.23`, avoiding overlong prompts for the next runtime comparison.

## Truth Prefill Logits 1000 Short From prompt10000 (2026-06-24)

- Path: `experiments/gemma4_IO_behaviors/mem8G/statistics/truth_prefill_logits_1000_short_from_prompt10000_20260624`
- Purpose: larger prompt set for live RPP comparisons while keeping prompt lengths bounded. This reuses existing `dataset/prompt10000` router-label NPZ files; no new decode or logits dump was run.
- Source layout compatibility: produced the same live-RPP truth-root layout used by previous short runs: `selected_prompt_database.jsonl`, `router_label_npz/dump_pack_manifest.csv`, `router_label_npz/npz/*.npz`, `generations/generations_manifest.jsonl`, `generations/completions/*.json`, and prefill-only `oracle_live_trace.jsonl`.
- Selection: source-balanced shortest samples from `dataset/prompt10000`, preserving the original 300/200/200/200/100 mix. Token lengths were taken from `dataset/prompt10000/router_label_npz/dump_pack_manifest.csv`; requested cap was `max_total_tokens=192`.
- Source counts: `tatsu-lab/alpaca=300`, `EdinburghNLP/xsum=200`, `wmt/wmt16=200`, `flwrlabs/code-alpaca-20k=200`, `EleutherAI/hendrycks_math=100`.
- Token stats: samples `1,000`, tokens total `49,233`, min `22`, mean `49.233`, max `153`, loss tokens total `9,912`. The selected set stayed below the requested cap.
- Per-source token max: alpaca `27`, xsum `153`, wmt `41`, code `30`, math `52`.
- Storage: `1,000` NPZ files and `1,000` completion JSON files were hardlinked into the truth root, avoiding duplicate large-file copies. Linked NPZ byte total from manifest: `325,763,946`.
- Oracle trace: generated with `build_oracle_live_trace.py --include-prefill --no-decode`; trace rows `39,321`, sample count `1,000`, phases `{'prefill': 39321}`.
- Validation: `validate_truth_prefill_subset.py` passed with `errors=0`; prompt order matched manifest sample order, all NPZ paths existed, trace rows matched `tokens_total - loss_tokens_total`, and no decode rows were present.
- To use in live RPP runs: set `TRUTH_ROOT=experiments/gemma4_IO_behaviors/mem8G/statistics/truth_prefill_logits_1000_short_from_prompt10000_20260624` and use `PROMPT_LIMIT` up to `1000`.
