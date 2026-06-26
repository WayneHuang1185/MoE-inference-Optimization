# Local IO Experiment Summary

Source: `phase2_gpu_topk_sweep_selected.jsonl`

| scenario | requests | prompts | mean TTFT | mean total | mean tok/s | mean major faults | mean minor faults | mean SwapFree delta KB | mean SwapCached delta KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-gpu-topk2 | 60 | 20 | 2.911 | 5.933 | 5.313 | 88135 | 180690 | -9169 | 5744 |
| rpp-first-token-gpu-topk4 | 60 | 20 | 2.984 | 6.029 | 5.295 | 73776 | 178352 | 7412 | 6980 |
| rpp-first-token-gpu-topk8 | 60 | 20 | 2.991 | 5.828 | 5.489 | 105538 | 154873 | -75106 | 4403 |
| two-step-no-rpp-gpu | 60 | 20 | 3.057 | 4.991 | 6.424 | 122608 | 169014 | -96529 | 5769 |
