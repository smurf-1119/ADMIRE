set -x

# CHECKPOINT=/gruntdata/heyuan2/workspace/bobblair.wj/zqp/qp_lm_models/mpdocvqa/qwen2-vl-7b-instruct/v6-20241108-144520/checkpoint-4484/
CHECKPOINT=$1
DATASET=$2
GPUS=$3

CHECKPOINT="$(pwd)/${CHECKPOINT}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=${MASTER_PORT:-62621}
PORT=${PORT:-62621}
GPUS=$GPUS
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}


if [ ${DATASET} == "vqa-mpdocvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-100352" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --max-pixels 100352 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets newsvqa_val --out-dir ./res/news-vqa ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-100352" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets newsvqa_val --out-dir ./res/news-vqa --max-pixels 100352 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-dynamic" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-dynamic" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-dynamic" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls2.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v3" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v3-top3" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --dynamic --select-mode topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v3-top5" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --dynamic --select-mode topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v3-dynamic" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --dynamic --select-mode dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls2.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3-100352-1003520" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode topk --select-image-num 3 --max-pixels 100352 --max-pixels-all 1003520 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5-100352-1003520" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode topk --select-image-num 5 --max-pixels 100352 --max-pixels-all 1003520 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v3-dynamic" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/mpdoc-vqa --dynamic --select-mode dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v3" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v3-dynamic" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v3-top3" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v3-top5" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v4-dynamic_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --dynamic --select-mode dynamic_mix ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --out-dir ./res/mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls3.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode dynamic_mix ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4_top3_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix-200704-602112" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 5 --max-pixels 200704 --max-pixels-all 602112 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix-200704-401408" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 5 --max-pixels 200704 --max-pixels-all 401408 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix-200704-401408" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 3 --max-pixels 200704 --max-pixels-all 401408 ${@:4}
fi


if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-unsupervised-cls-v4-dynamic_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --out-dir ./res/dude-mpdoc-vqa --dynamic --select-mode mix_dynamic --max-pixels-all 4011408 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v4-top3_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode mix_topk --select-image-num 3 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v4-top5_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode mix_topk --select-image-num 5 ${@:4}
fi

if [ ${DATASET} == "vqa-newsvqa-val-unsupervised-cls-v4-dynamic_mix" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic_unsupervised_cls4.py --checkpoint ${CHECKPOINT} --datasets newsvqa_val --out-dir ./res/news-vqa --dynamic --select-mode mix_dynamic ${@:4}
fi

if [ ${DATASET} == "vqa-mpdocvqa-val-allpixels" ]; then
for MAX_PIXELS in 50176 100352 150528 200704 250880 301056 351232 401408
do
    echo 'MAX_PIXELS:'$MAX_PIXELS
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --max-pixels $MAX_PIXELS --out-dir ./res/mpdoc-vqa ${@:4}
wait
done
fi

if [ ${DATASET} == "vqa-dude-mpdocvqa-val-allpixels" ]; then
for MAX_PIXELS in 100352
do
    echo 'MAX_PIXELS:'$MAX_PIXELS
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    infer_mpdocvqa_dynamic.py --checkpoint ${CHECKPOINT} --datasets dude_mpdocvqa_val --max-pixels $MAX_PIXELS --out-dir ./res/dude-mpdoc-vqa ${@:4}
wait
done
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
