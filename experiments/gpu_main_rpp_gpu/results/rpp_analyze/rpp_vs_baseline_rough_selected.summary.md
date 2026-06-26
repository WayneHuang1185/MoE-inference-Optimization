# Local IO Experiment Summary

Source: `rpp_vs_baseline_rough_selected.jsonl`

| scenario | requests | prompts | mean TTFT | mean total | mean tok/s | mean major faults | mean minor faults | mean SwapFree delta KB | mean SwapCached delta KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-cpu-cold | 60 | 20 | 5.385 | 32.393 | 0.962 | 624187 | 1807404 | nan | nan |
| baseline-gpu-cpu-moe | 60 | 20 | 1.356 | 2.736 | 12.383 | 16373 | 63784 | -21082 | -326 |
| baseline-gpu-ncpu-moe-32 | 60 | 20 | 0.882 | 1.892 | 17.464 | 8952 | 40969 | -20204 | -761 |
| rpp-first-token-cpu | 60 | 20 | 5.433 | 16.164 | 1.936 | 210662 | 649397 | -10458 | -3418 |
| rpp-first-token-gpu | 60 | 20 | 2.991 | 5.828 | 5.489 | 105538 | 154873 | -75106 | 4403 |
