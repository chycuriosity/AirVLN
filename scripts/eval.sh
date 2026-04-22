if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate cxj
fi

cd ./AirVLN
echo $PWD

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CKPT_PATH=${LOCAL_EVAL_CKPT_PATH:-/data/lyj/cxj/AirVLN_ws/DATA/output/AirVLN-seq2seq/train/checkpoint/20251123-101135-680949/ckpt.LAST.pth}
EVAL_DATASET=${LOCAL_EVAL_DATASET:-val_unseen}
EVAL_NUM=${LOCAL_EVAL_NUM:--1}
MAX_ACTION=${LOCAL_EVAL_MAX_ACTION:-100}
BATCH_SIZE=${LOCAL_EVAL_BATCH_SIZE:-1}
GPU_DEVICE=${LOCAL_EVAL_GPU_DEVICE:-2}

python -u ./src/vlnce_src/train.py \
--run_type eval \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize "$BATCH_SIZE" \
--trainer_gpu_device "$GPU_DEVICE" \
--EVAL_CKPT_PATH_DIR "$CKPT_PATH" \
--EVAL_DATASET "$EVAL_DATASET" \
--EVAL_NUM "$EVAL_NUM" \
--maxAction "$MAX_ACTION"
