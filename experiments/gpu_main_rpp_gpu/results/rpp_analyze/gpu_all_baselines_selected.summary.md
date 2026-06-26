# Local IO Experiment Summary

Source: `gpu_all_baselines_selected.jsonl`

| scenario | requests | prompts | mean TTFT | mean total | mean tok/s | mean major faults | mean minor faults | mean SwapFree delta KB | mean SwapCached delta KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-gpu-cpu-moe | 60 | 20 | 1.356 | 2.736 | 12.383 | 16373 | 63784 | -21082 | -326 |
| baseline-gpu-ncpu-moe-32 | 60 | 20 | 0.882 | 1.892 | 17.464 | 8952 | 40969 | -20204 | -761 |
| baseline-gpu-ngl10 | 60 | 20 | 1.262 | 3.105 | 10.179 | 10612 | 46363 | -21760 | -668 |
