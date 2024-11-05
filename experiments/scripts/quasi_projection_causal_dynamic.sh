#!/bin/bash

# Define shared variables
MODEL="meta-llama/Meta-Llama-3-8B"
SAVE_DIR="./assets/checkpoints"
SUBSPACE_MODULE="QuasiProjective"
N_EPOCHS=1
N_STEPS=500
LAMBDA_PARAMETER=1e-3
EVAL_PER_STEPS=100
TRAIN_BATCH_SIZE=32

TRAIN_PATH="experiments/RAVEL/data/city_train"
TEST_PATH="experiments/RAVEL/data/city_test"

DICTIONARY_SIZE=32
SCORING_DIMENSION=2,4,8,16,32
FREEZE_DAS_MODULES=null

LOG_WANDB=true
WANDB_PROJECT="HyperDAS"
WANDB_GROUP="quasi_dynamic_selection"
WANDB_ENTITY="hyperdas"
WANDB_NOTES="quasi-comparison-city-full"

LOAD_FROM=null
ORTHOGONAL_INIT=true

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
    log_wandb=$LOG_WANDB \
    wandb_group=$WANDB_GROUP \
    debug_model=true \
    train_batch_size=$TRAIN_BATCH_SIZE \
    train_path=$TRAIN_PATH \
    test_path=$TEST_PATH \
    dict_size=$DICTIONARY_SIZE \
    wandb_notes="$WANDB_NOTES" \
    return_penalty=false \
    orthogonal_init=$ORTHOGONAL_INIT \
    selection_mechanism=dynamic \
    scoring_dimension=$SCORING_DIMENSION \
    freeze_das_module=$FREEZE_DAS_MODULES \
    load_trained_from=$LOAD_FROM \
    save_model=true \
    hydra.launcher.n_jobs=2