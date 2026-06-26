# Phase 2 RPP Breakdown Summary

Source: `phase2_gpu_rpp_vs_no_rpp_selected.jsonl`

| scenario | requests | prompts | total s | first token s | continuation s | tokenize ms | RPP ms | prefetch ms | prefetch MB | major faults | first major | continuation major |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-gpu | 60 | 20 | 5.828 | 2.994 | 1.752 | 787.0 | 147.4 | 148.4 | 487 | 105538 | 88850 | 16688 |
| two-step-no-rpp-gpu | 60 | 20 | 4.991 | 3.060 | 1.931 | 0.0 | 0.0 | 0.0 | 0 | 122608 | 99600 | 23009 |
