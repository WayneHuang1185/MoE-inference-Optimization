# Local IO Experiment Summary

Source: `phase2_gpu_rpp_vs_no_rpp_selected.jsonl`

| scenario | requests | prompts | mean TTFT | mean total | mean tok/s | mean major faults | mean minor faults | mean SwapFree delta KB | mean SwapCached delta KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-gpu | 60 | 20 | 2.991 | 5.828 | 5.489 | 105538 | 154873 | -75106 | 4403 |
| two-step-no-rpp-gpu | 60 | 20 | 3.057 | 4.991 | 6.424 | 122608 | 169014 | -96529 | 5769 |
