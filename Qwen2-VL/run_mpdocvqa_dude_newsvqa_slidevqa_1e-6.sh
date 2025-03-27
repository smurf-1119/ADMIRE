mkdir /mntnlp
mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-23-tpc3.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

mkdir /model_hub
mount -t nfs -o vers=3,nolock,proto=tcp alipay-heyuan-12-nvy33.cn-heyuan-alipay.nas.aliyuncs.com:/ /model_hub

mkdir /mnt/prev_nas
mount -t nfs -o vers=3,nolock,proto=tcp alipay-heyuan-10-emx24.cn-heyuan-alipay.nas.aliyuncs.com:/ /mnt/prev_nas/

mkdir -p /gruntdata/heyuan2 
mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipayheyuan2-12-clc63.cn-heyuan-alipay.nas.aliyuncs.com:/ /gruntdata/heyuan2

mkdir /gruntdata/heyuan67 
mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-67-sgk32.cn-heyuan-alipay.nas.aliyuncs.com:/ /gruntdata/heyuan67

cd /mntnlp/zqp/physical_report/Qwen2-VL
echo "running"

MAX_RESOLUTION=448 NPROC_PER_NODE=8 NNODES=2 NODE_RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} swift sft \
  --model_type qwen2-vl-7b-instruct \
  --model_id_or_path /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct \
  --sft_type full \
  --dataset /gruntdata/heyuan67/zqp/MIX_MPDOC_DUDE_NEWSVQA/qwen2vl_swift_train_mpdocvqa_dude_newsvqa_slidevqa.jsonl \
  --deepspeed ds_z3_offload_config.json \
  --target_modules AUTO \
  --learning_rate 1e-6 \
  --gradient_accumulation_steps 1 \
  --freeze_vit False \
  --logging_steps 1 \
  --batch_size 1 \
  --output_dir /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa_max_resolution_448/1e-6 \
  --dataloader_num_workers 8 \
  --save_safetensors True \
  --save_only_model True \
  --gradient_checkpointing True \
  --save_strategy epoch \
  --push_to_hub False \
  --use_flash_attn True \
  --dataset_test_ratio 0. \
  --num_train_epochs 2 \
  --save_total_limit 2 \