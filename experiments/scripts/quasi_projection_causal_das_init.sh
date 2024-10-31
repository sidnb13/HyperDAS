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
TRAIN_PATH="experiments/RAVEL/data/ravel_country_causal_only_train"
TEST_PATH="experiments/RAVEL/data/ravel_country_causal_only_test"
DICTIONARY_SIZE=4096
TOP_K_PARAMETER=2,4,8,16,32,512,1024,2048,3072

WANDB_PROJECT="HyperDAS"
WANDB_GROUP="quasi_test_simple"
WANDB_ENTITY="hyperdas"
WANDB_NOTES=null

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
    top_k_parameter=$TOP_K_PARAMETER \
    dict_size=$DICTIONARY_SIZE \
    ridge_parameterization=topk_ste \
    wandb_notes="$WANDB_NOTES" \
    return_penalty=true \
    do_topk=true \
    orthogonal_init=true \
    hydra.launcher.n_jobs=2