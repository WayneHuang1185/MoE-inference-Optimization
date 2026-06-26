# Phase 2 RPP Breakdown Summary

Source: `phase2_cpu_rpp_vs_no_rpp_selected.jsonl`

| scenario | requests | prompts | total s | first token s | continuation s | tokenize ms | RPP ms | prefetch ms | prefetch MB | major faults | first major | continuation major |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rpp-first-token-cpu | 60 | 20 | 16.164 | 5.436 | 8.956 | 872.1 | 181.8 | 718.3 | 612 | 210662 | 84018 | 126645 |
| two-step-no-rpp-cpu | 60 | 20 | 12.666 | 5.454 | 7.212 | 0.0 | 0.0 | 0.0 | 0 | 175989 | 86838 | 89151 |
