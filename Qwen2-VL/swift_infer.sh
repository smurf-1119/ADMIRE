# ps -ef | grep swift | grep -v grep | awk '{print "sudo kill -9 "$2}' | sh
MAX_PIXELS=100352 NPROC_PER_NODE=8 swift infer \
    --ckpt_dir model_weights/v5-20241117-233245/checkpoint-13587 \
    --use_flash_attn false \
    --val_dataset /mntnlp/qp_mm_data/docvqa/mpdoc/mpdoc_val_swift_qwen2vl.jsonl