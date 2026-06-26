# Phase 2 RPP Breakdown Summary

Source: `phase2_gpu_topk_sweep_selected.jsonl`

| scenario | requests | prompts | total s | first token s | continuation s | tokenize ms | RPP ms | prefetch ms | prefetch MB | major faults | first major | continuation major |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-gpu-topk2 | 60 | 20 | 5.933 | 2.915 | 2.021 | 798.5 | 156.5 | 41.7 | 122 | 88135 | 62401 | 25735 |
| rpp-first-token-gpu-topk4 | 60 | 20 | 6.029 | 2.987 | 1.984 | 807.3 | 167.2 | 83.5 | 243 | 73776 | 49982 | 23794 |
| rpp-first-token-gpu-topk8 | 60 | 20 | 5.828 | 2.994 | 1.752 | 787.0 | 147.4 | 148.4 | 487 | 105538 | 88850 | 16688 |
| two-step-no-rpp-gpu | 60 | 20 | 4.991 | 3.060 | 1.931 | 0.0 | 0.0 | 0.0 | 0 | 122608 | 99600 | 23009 |
