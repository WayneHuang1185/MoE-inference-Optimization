# Package Manifest

Generated for continuing Gemma4 Global RoutingPathPredictor training on Vast.ai.

Required payload:

- Training code copied from `experiments/gemma4_global_predictor`.
- Grid search runner copied from `experiments/gemma4_global_predictor/grid_search_rpp.py`.
- Dataset copied from `dataset/prompt10000/router_label_npz/npz`.
- Current checkpoint snapshot copied from `experiments/gemma4_bottleneck/results/rpp_train_stratified_prompt10000_fold0_cpu32_bestresume30_shuffle_20260515_190609/checkpoint_best.pt`.
- Stable resume checkpoint copied from `experiments/gemma4_bottleneck/results/rpp_train_stratified_prompt10000_fold0_cpu32_bestresume20_20260515_1325/checkpoint_best.pt`.

The active remote run at packaging time used:

```text
python3 -m experiments.gemma4_global_predictor.train_rpp
  --data-root dataset/prompt10000/router_label_npz/npz
  --out-dir experiments/gemma4_bottleneck/results/rpp_train_stratified_prompt10000_fold0_cpu32_bestresume30_shuffle_20260515_190609
  --max-files 0
  --max-seq-len 512
  --epochs 30
  --batch-size 8
  --lr 3e-4
  --weight-decay 0.01
  --grad-clip 1.0
  --num-workers 4
  --seed 0
  --shuffle-seed -1
  --device auto
  --vocab-size 262144
  --embedding-mode hash
  --hash-vocab-size 32768
  --d-model 32
  --n-heads 4
  --encoder-layers 2
  --decoder-layers 2
  --ffn-dim 2048
  --dropout 0.1
  --pos-weight 15.0
  --bce-weight 1.0
  --kl-weight 0.0
  --kl-schedule fixed
  --kl-start 0.05
  --kl-end 0.5
  --kl-warmup-ratio 0.35
  --temperature 1.0
  --log-every 25
  --resume checkpoints/current/checkpoint_best.pt
  --split-strategy stratified
  --train-frac 0.8
  --val-folds 5
  --val-fold-index 0
  --stratify-keys task_type,source
```

The grid-search Vast entrypoint is:

```text
./run_vast_grid_search.sh
```

It ranks candidate runs by validation `batch_level_accuracy@8`.
