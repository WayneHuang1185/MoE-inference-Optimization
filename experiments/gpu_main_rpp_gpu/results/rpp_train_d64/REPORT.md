# Qwen3.6 Global RPP Training Report

## Config

```json
{
  "batch_size": 16,
  "bce_weight": 1.0,
  "d_model": 64,
  "data_root": "../npz/",
  "data_summary": {
    "all": {
      "files": 10000,
      "loss_tokens": 978402,
      "max_token_id": 248069,
      "seq_len_gt_512": 33,
      "source_counts": {
        "Rowan/hellaswag": 2000,
        "cais/mmlu": 3000,
        "google-research-datasets/mbpp": 836,
        "openai/gsm8k": 2000,
        "openai_humaneval": 164,
        "wikitext": 2000
      },
      "task_counts": {
        "code_generation": 1000,
        "commonsense": 2000,
        "math_reasoning": 2000,
        "multiple_choice": 3000,
        "text_continuation": 2000
      },
      "tokens": 1794962
    },
    "test": {
      "files": 1599,
      "loss_tokens": 156197,
      "max_token_id": 248069,
      "seq_len_gt_512": 3,
      "source_counts": {
        "Rowan/hellaswag": 320,
        "cais/mmlu": 480,
        "google-research-datasets/mbpp": 133,
        "openai/gsm8k": 320,
        "openai_humaneval": 26,
        "wikitext": 320
      },
      "task_counts": {
        "code_generation": 159,
        "commonsense": 320,
        "math_reasoning": 320,
        "multiple_choice": 480,
        "text_continuation": 320
      },
      "tokens": 285868
    },
    "train": {
      "files": 8000,
      "loss_tokens": 783362,
      "max_token_id": 248069,
      "seq_len_gt_512": 29,
      "source_counts": {
        "Rowan/hellaswag": 1600,
        "cais/mmlu": 2400,
        "google-research-datasets/mbpp": 669,
        "openai/gsm8k": 1600,
        "openai_humaneval": 131,
        "wikitext": 1600
      },
      "task_counts": {
        "code_generation": 800,
        "commonsense": 1600,
        "math_reasoning": 1600,
        "multiple_choice": 2400,
        "text_continuation": 1600
      },
      "tokens": 1438687
    },
    "val": {
      "files": 401,
      "loss_tokens": 38843,
      "max_token_id": 248069,
      "seq_len_gt_512": 1,
      "source_counts": {
        "Rowan/hellaswag": 80,
        "cais/mmlu": 120,
        "google-research-datasets/mbpp": 34,
        "openai/gsm8k": 80,
        "openai_humaneval": 7,
        "wikitext": 80
      },
      "task_counts": {
        "code_generation": 41,
        "commonsense": 80,
        "math_reasoning": 80,
        "multiple_choice": 120,
        "text_continuation": 80
      },
      "tokens": 70407
    }
  },
  "decoder_layers": 4,
  "device": "auto",
  "device_resolved": "cuda",
  "dropout": 0.1,
  "embedding_mode": "hash",
  "encoder_layers": 4,
  "epochs": 150,
  "experts": 256,
  "ffn_dim": 256,
  "grad_clip": 1.0,
  "hash_vocab_size": 32768,
  "head_hidden_dim": 0,
  "kl_end": 0.5,
  "kl_schedule": "linear",
  "kl_start": 0.05,
  "kl_warmup_ratio": 0.35,
  "kl_weight": 0.2,
  "layers": 40,
  "log_every": 10,
  "lr": 0.0003,
  "max_files": 0,
  "max_seq_len": 512,
  "n_heads": 4,
  "negative_experts_per_layer": 248,
  "num_workers": 4,
  "out_dir": "results/rpp_train_d64",
  "parameter_count": 3295360,
  "parameter_count_by_component": {
    "output_heads": 665600,
    "token_embedding": 2097152,
    "token_embedding_table_size": 32768,
    "total": 3295360,
    "transformer_and_positions": 532608
  },
  "pos_weight": 31.0,
  "positive_experts_per_layer": 8,
  "positive_fraction": 0.03125,
  "resume": "",
  "resume_epoch": 0,
  "seed": 0,
  "shuffle_seed": -1,
  "shuffle_seed_resolved": 0,
  "split_config": {
    "holdout_frac": 0.19999999999999996,
    "strategy": "stratified",
    "stratify_keys": [
      "task_type",
      "source"
    ],
    "train_frac": 0.8,
    "val_fold_index": 0,
    "val_folds": 5
  },
  "split_counts": {
    "test": 1599,
    "train": 8000,
    "val": 401
  },
  "split_strategy": "stratified",
  "start_epoch": 1,
  "stratify_keys": "task_type,source",
  "temperature": 1.0,
  "top_k": 8,
  "torch_num_threads": 64,
  "total_train_steps": 75000,
  "train_frac": 0.8,
  "val_fold_index": 0,
  "val_folds": 5,
  "vocab_size": 249093,
  "weight_decay": 0.01
}
```

## Final Metrics

| phase | loss | bce | kl | kl_weight | weighted_kl | token_recall@8 | token_recall@16 | token_top1 | token_exact@8 | batch_level_accuracy@8 | batch_level_accuracy@16 | kl_true_pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.604672 | 0.462644 | 0.285366 | 0.497697 | 0.142028 | 0.736887 | 0.863621 | 0.445855 | 0.001350 | 0.927158 | 0.976899 | 0.285366 |
| val | 0.628128 | 0.478689 | 0.298878 | 0.500000 | 0.149439 | 0.711645 | 0.838154 | 0.432569 | 0.001333 | 0.848480 | 0.945805 | 0.298878 |
| test | 0.626941 | 0.477448 | 0.298987 | 0.500000 | 0.149493 | 0.713578 | 0.839318 | 0.432746 | 0.001413 | 0.847418 | 0.944707 | 0.298987 |

## Best Validation

- epoch: 66
- val batch_level_accuracy@8: 0.854682
- val loss: 0.464055
- checkpoint: `results/rpp_train_d64/checkpoint_best.pt`
