SHARED_ARGS="--split RANDOM \
    --split-sizes 0.8 0.2 0.0 \
    --num-replicates 1 \
    --data-seed 0 \
    --pytorch-seed 0 \
    --loss-function mse \
    --metrics rmse mae \
    --task-type regression \
    --ffn-num-layers 0 \
    --smiles-columns SmilesCurated \
    --target-columns ExperimentalLogS \
    --data-path data/AqSolDBc.csv"

# chemprop train \
#     --ffn-hidden-dim 2100 \
#     --aggregation mean \
#     --activation tanh \
#     --depth 6 \
#     --message-hidden-dim 1000 \
#     $SHARED_ARGS

# chemprop hpopt \
#     --search-parameter-keywords activation aggregation depth message_hidden_dim ffn_hidden_dim \
#     --hpopt-save-dir chemprop_hpopt \
#     --hyperopt-random-state-seed 0 \
#     --raytune-num-samples 128 \
#     --raytune-search-algorithm optuna \
#     --raytune-num-gpus 8 \
#     --raytune-max-concurrent-trials 8 \
#     --raytune-use-gpu \
#     $SHARED_ARGS

# run on one GPU to preserve order for sure
CUDA_VISIBLE_DEVICES=0 chemprop fingerprint \
    --test-path data/OChemUnseen_valid.csv \
    --output data/test_chemprop_fprints.csv \
    --model-path chemprop_training/AqSolDBc/2025-01-27T10-18-38/model_0/best.pt \
    --ffn-block-index 0 \
    --smiles-columns SMILES