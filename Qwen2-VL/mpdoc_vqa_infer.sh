set -x

# CHECKPOINT=/gruntdata/heyuan2/workspace/bobblair.wj/zqp/qp_lm_models/mpdocvqa/qwen2-vl-7b-instruct/v6-20241108-144520/checkpoint-4484/
CHECKPOINT=$1
DATASET=$2

CHECKPOINT="$(pwd)/${CHECKPOINT}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=${MASTER_PORT:-62621}
PORT=${PORT:-62621}
GPUS=8
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

if [ ${DATASET} == "vqa-mpdocvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa ${@:3}
fi

if [ ${DATASET} == "vqa-mpdocvqa-train1000" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_train1000 --out-dir ./res/mpdoc-vqa ${@:3}
fi
