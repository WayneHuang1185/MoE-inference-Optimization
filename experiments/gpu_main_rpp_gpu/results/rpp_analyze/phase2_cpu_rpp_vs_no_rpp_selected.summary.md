# Local IO Experiment Summary

Source: `phase2_cpu_rpp_vs_no_rpp_selected.jsonl`

| scenario | requests | prompts | mean TTFT | mean total | mean tok/s | mean major faults | mean minor faults | mean SwapFree delta KB | mean SwapCached delta KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-cpu | 60 | 20 | 5.433 | 16.164 | 1.936 | 210662 | 649397 | -10458 | -3418 |
| two-step-no-rpp-cpu | 60 | 20 | 5.452 | 12.666 | 2.499 | 175989 | 589653 | -13113 | 541 |
