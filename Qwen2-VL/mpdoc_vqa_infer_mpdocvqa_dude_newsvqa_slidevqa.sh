set -x

# CHECKPOINT=/gruntdata/heyuan2/workspace/bobblair.wj/zqp/qp_lm_models/mpdocvqa/qwen2-vl-7b-instruct/v6-20241108-144520/checkpoint-4484/
CHECKPOINT=$1
DATASET=$2
GPUS=$3

CHECKPOINT="${CHECKPOINT}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
PORT=${PORT:-62621}
GPUS=$GPUS
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

########################### mpdocvqa ###########################
if [ ${DATASET} == "vqa-mpdocvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --max-pixels 200704 --out-dir  ./res/mpdoc-vqa ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v3-top3" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --select-mode topk --max-pixels-low 200704 --max-pixels-high 802816 --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v3-top3-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --select-mode topk --max-pixels-low 200704 --max-pixels-high 401408 --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v3-top5" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --select-mode topk --max-pixels-low 200704 --max-pixels-high 802816 --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v3-top5-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --select-mode topk --max-pixels-low 200704 --max-pixels-high 401408 --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --max-pixels-low 200704 --max-pixels-high 802816 --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --max-pixels-low 200704 --max-pixels-high 401408 --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --max-pixels-low 200704 --max-pixels-high 802816 --select-mode mix_topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --max-pixels-low 200704 --max-pixels-high 401408 --select-mode mix_topk --select-image-num 5 ${@:4}
fi

########################### mpdocvqa ###########################

########################### DUDE ###########################
if [ ${DATASET} == "vqa-dude-mpdocvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --max-pixels 200704 --out-dir ./res/dude-mpdoc-vqa ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --max-pixels-low 200704 --max-pixels-high 802816 --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --max-pixels-low 200704 --max-pixels-high 401408 --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --max-pixels-low 200704 --max-pixels-high 802816 --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --max-pixels-low 200704 --max-pixels-high 401408 --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --max-pixels-low 200704 --max-pixels-high 802816 --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --max-pixels-low 200704 --max-pixels-high 401408 --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --max-pixels-low 200704 --max-pixels-high 802816 --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --max-pixels-low 200704 --max-pixels-high 401408 --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 5 ${@:4}
fi
########################### DUDE ###########################

########################### newsvqa ###########################
if [ ${DATASET} == "vqa-newsvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --max-pixels 200704 --datasets newsvqa_val --out-dir ./res/news-vqa ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v3-top3" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --max-pixels-low 200704 --max-pixels-high 802816 --datasets newsvqa_val --out-dir ./res/news-vqa --select-mode topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v3-top3-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --max-pixels-low 200704 --max-pixels-high 401408 --datasets newsvqa_val --out-dir ./res/news-vqa --select-mode topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v3-top5" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --max-pixels-low 200704 --max-pixels-high 802816 --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v3-top5-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --max-pixels-low 200704 --max-pixels-high 401408 --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v4-top3_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT}  --max-pixels-low 200704 --max-pixels-high 802816 --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v4-top3_mix-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT}  --max-pixels-low 200704 --max-pixels-high 401408 --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v4-top5_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT}  --max-pixels-low 200704 --max-pixels-high 802816 --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode mix_topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v4-top5_mix-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT}  --max-pixels-low 200704 --max-pixels-high 401408 --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode mix_topk --select-image-num 5 ${@:4}
fi
########################### newsvqa ###########################

########################### slidevqa ###########################
if [ ${DATASET} == "vqa-slidevqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --max-pixels 200704 --datasets slidevqa_val --out-dir ./res/slidevqa-vqa ${@:4}
fi

if [ ${DATASET} == "vqa-slidevqa-val-unsupervised-cls-v3-top3" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --max-pixels-low 200704 --max-pixels-high 802816 --datasets slidevqa_val --out-dir ./res/slidevqa-vqa --select-mode topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-slidevqa-val-unsupervised-cls-v3-top3-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --max-pixels-low 200704 --max-pixels-high 401408 --datasets slidevqa_val --out-dir ./res/slidevqa-vqa --select-mode topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-slidevqa-val-unsupervised-cls-v3-top5" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --max-pixels-low 200704 --max-pixels-high 802816 --datasets slidevqa_val --out-dir ./res/slidevqa-vqa --dynamic --select-mode topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-slidevqa-val-unsupervised-cls-v3-top5-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --max-pixels-low 200704 --max-pixels-high 401408 --datasets slidevqa_val --out-dir ./res/slidevqa-vqa --dynamic --select-mode topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-slidevqa-val-unsupervised-cls-v4-top3_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT}  --max-pixels-low 200704 --max-pixels-high 802816 --datasets slidevqa_val --out-dir ./res/slide-vqa --dynamic --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-slidevqa-val-unsupervised-cls-v4-top3_mix-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT}  --max-pixels-low 200704 --max-pixels-high 401408 --datasets slidevqa_val --out-dir ./res/slide-vqa --dynamic --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-slidevqa-val-unsupervised-cls-v4-top5_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT}  --max-pixels-low 200704 --max-pixels-high 802816 --datasets slidevqa_val --out-dir ./res/slide-vqa --dynamic --select-mode mix_topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-slidevqa-val-unsupervised-cls-v4-top5_mix-test2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT}  --max-pixels-low 200704 --max-pixels-high 401408 --datasets slidevqa_val --out-dir ./res/slide-vqa --dynamic --select-mode mix_topk --select-image-num 5 ${@:4}
fi

########################### slidevqa ###########################

