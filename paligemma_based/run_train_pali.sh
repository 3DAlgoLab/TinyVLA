export MODEL_NAME=paligemma-3b-pt-224
export CKPT_FILE=paligemma-3b-pt-224.npz

env BV_GEMMA_DIR=ckpts/ python -m big_vision.trainers.proj.paligemma.train \
    --config bv_pali_cfg.py \
    --workdir workdirs/`date '+%m-%d_%H%M'`