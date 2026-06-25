# Cross-Memory Inference Boundness

Runs the same 30 short prompts under multiple Docker memory caps and records
latency, CPU parallelism, cgroup `pgmajfault`, storage read MB, and PSI
I/O/memory stall deltas.

```bash
ssh nthu-cs 'cd ~/workspace/HuangWayne/project && MEMORY_CAPS=24g,10g,8g,6g N_PREDICT=16 PROMPT_LIMIT=30 MAX_PROMPT_CHARS=256 experiments/gemma4_IO_behaviors/cross_memory_boundness/run_cross_memory_boundness.sh'
```

Outputs are written under:

```text
experiments/gemma4_IO_behaviors/cross_memory_boundness/statistics/cross_memory_boundness_<timestamp>/
```
