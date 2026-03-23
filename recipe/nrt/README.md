<!--
Copyright 2026 Yuanfu Wang
Modified by Yuanfu Wang (Shanghai Artificial Intelligence)

Licensed under the Apache License, Version 2.0 (the "License");
-->

# NRT Recipe (verl)

This folder contains a minimal recipe to run NRT-style training in `verl`.

## 1. Add required special token(s) to the base model

The NRT pipeline uses prompt/segment markers such as:

- `<|response_start|>` (added by this repo)

Add `<|response_start|>` to your original model by resizing tokenizer embeddings and initializing the new token from `eos_token`:

```bash
python3 recipe/nrt/add_response_start_token_to_model.py \
  --model_path /path/to/base_model \
  --output_path /path/to/base_model_with_response_start
```

The output model directory will be used as `actor_rollout_ref.model.path`.

## 2. Prepare the dataset (parquet)

You need train/val parquet files containing (at least) the columns produced by:

- `recipe/nrt/data/scripts/tulu-3-sft-mixture.py`

Example (Tulu-3 SFT mixture preprocessing):

```bash
python3 recipe/nrt/data/scripts/tulu-3-sft-mixture.py \
  --input_dir ./recipe/nrt/data/raw/tulu-3-sft-mixture/ \
  --output_dir ./recipe/nrt/data/processed/tulu-3-sft-mixture-processed \
  --train_size 220000 \
  --test_size 1024 \
  --num_proc 100
```

This creates:

- `.../train.parquet`
- `.../test.parquet`

## 3. Run training

The recommended way is to follow the example script `recipe/nrt/run_example_log_prob.sh` (it contains the full Hydra overrides).

Before running, set at least:

- `MODEL_PATH` (your modified model from step 1)
- `MLP_WORKER_GPU` and `MLP_WORKER_NUM` (used to set `trainer.n_gpus_per_node` and `trainer.nnodes`)

Example:

```bash
export MODEL_PATH=/path/to/base_model_with_response_start
export MLP_WORKER_GPU=1
export MLP_WORKER_NUM=1

bash recipe/nrt/run_example_log_prob.sh
```

