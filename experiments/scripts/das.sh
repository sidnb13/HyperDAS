#!/bin/bash

# Define shared variables
MODEL="meta-llama/Meta-Llama-3-8B"
SAVE_DIR="./assets/checkpoints"
SUBSPACE_MODULE="ReflectSelect"
N_EPOCHS=1
N_STEPS=1000
LAMBDA_PARAMETER=1e-3
EVAL_PER_STEPS=100
TRAIN_BATCH_SIZE=16
DAS_DIMENSION=2,4,8,16,32,64,128
TRAIN_PATH="experiments/RAVEL/data/ravel_country_causal_only_train"
TEST_PATH="experiments/RAVEL/data/ravel_country_causal_only_test"

WANDB_PROJECT="HyperDAS"
WANDB_GROUP="reflectdas_test_simple"
WANDB_ENTITY="hyperdas"
WANDB_NOTES="use simple causal dataset"

python train.py \
    --multirun \
    model_name_or_path=$MODEL \
    save_dir=$SAVE_DIR \
    subspace_module=$SUBSPACE_MODULE \
    n_epochs=$N_EPOCHS \
    compute_metrics=true \
    n_steps=$N_STEPS \
    lambda_parameter=$LAMBDA_PARAMETER \
    eval_per_steps=$EVAL_PER_STEPS \
    log_wandb=true \
    wandb_group=$WANDB_GROUP \
    debug_model=true \
    train_batch_size=$TRAIN_BATCH_SIZE \
    train_path=$TRAIN_PATH \
    test_path=$TEST_PATH \
    das_dimension=$DAS_DIMENSION \
    wandb_notes="$WANDB_NOTES" \
    return_penalty=false \
    hydra.launcher.n_jobs=4