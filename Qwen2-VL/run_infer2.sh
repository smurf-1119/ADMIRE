ps -ef | grep swift | grep -v grep | awk '{print "sudo kill -9 "$2}' | sh

# notrain 

## mpdocvqa
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-mpdocvqa-val 8 &> ./logs/notrain/vqa-mpdocvqa-val.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-mpdocvqa-val-unsupervised-cls-v3-top3-test2 6 &> ./logs/notrain/vqa-mpdocvqa-val-unsupervised-cls-v3-top3-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-mpdocvqa-val-unsupervised-cls-v3-top5-test2 8 &> ./logs/notrain/vqa-mpdocvqa-val-unsupervised-cls-v3-top5-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-mpdocvqa-val-unsupervised-cls-v3-top3 6 &> ./logs/notrain/vqa-mpdocvqa-val-unsupervised-cls-v3-top3.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-mpdocvqa-val-unsupervised-cls-v3-top5 8 &> ./logs/notrain/vqa-mpdocvqa-val-unsupervised-cls-v3-top5.log
# wait

# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix-test2 6 &> ./logs/notrain/vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix-test2 8 &> ./logs/notrain/vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix-test2.log
# wait

# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix 6 &> ./logs/notrain/vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix 8 &> ./logs/notrain/vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix.log
# wait

## dude
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-dude-mpdocvqa-val 8 &> ./logs/notrain/vqa-dude-mpdocvqa-val.log
# wait
# CUDA_VISIBLE_DEVICES=0 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3 1 &> ./logs/notrain/vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3.log &
# CUDA_VISIBLE_DEVICES=1 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5 1 &> ./logs/notrain/vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5.log &
CUDA_VISIBLE_DEVICES=0,1 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix 2 &> ./logs/notrain/vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix.log &
# CUDA_VISIBLE_DEVICES=3 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix 1 &> ./logs/notrain/vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix.log &

# CUDA_VISIBLE_DEVICES=4 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3-test2 1 &> ./logs/notrain/vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3-test2.log &
# CUDA_VISIBLE_DEVICES=5 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5-test2 1 &> ./logs/notrain/vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5-test2.log &
CUDA_VISIBLE_DEVICES=2,3 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix-test2 2 &> ./logs/notrain/vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix-test2.log &
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix-test2 4 &> ./logs/notrain/vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix-test2.log &
wait

# ## newsvqa
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-newsvqa-val 8 &> ./logs/notrain/vqa-newsvqa-val.log
# wait

bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-newsvqa-val-unsupervised-cls-v3-top3 8 &> ./logs/notrain/vqa-newsvqa-val-unsupervised-cls-v3-top3.log
wait
bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-newsvqa-val-unsupervised-cls-v3-top5 8 &> ./logs/notrain/vqa-newsvqa-val-unsupervised-cls-v3-top5.log
wait

bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-newsvqa-val-unsupervised-cls-v3-top3-test2 8 &> ./logs/notrain/vqa-newsvqa-val-unsupervised-cls-v3-top3-test2.log
wait
bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-newsvqa-val-unsupervised-cls-v3-top5-test2 8 &> ./logs/notrain/vqa-newsvqa-val-unsupervised-cls-v3-top5-test2.log
wait

# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-newsvqa-val-unsupervised-cls-v4-top3_mix 8 &> ./logs/notrain/vqa-newsvqa-val-unsupervised-cls-v4-top3_mix.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-newsvqa-val-unsupervised-cls-v4-top5_mix 8 &> ./logs/notrain/vqa-newsvqa-val-unsupervised-cls-v4-top5_mix.log
# wait

# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-newsvqa-val-unsupervised-cls-v4-top3_mix-test2 8 &> ./logs/notrain/vqa-newsvqa-val-unsupervised-cls-v4-top3_mix-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-newsvqa-val-unsupervised-cls-v4-top5_mix-test2 8 &> ./logs/notrain/vqa-newsvqa-val-unsupervised-cls-v4-top5_mix-test2.log
# wait

## slidevqa
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-slidevqa-val 8 &> ./logs/notrain/vqa-slidevqa-val.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-slidevqa-val-unsupervised-cls-v3-top3 8 &> ./logs/notrain/vqa-slidevqa-val-unsupervised-cls-v3-top3.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-slidevqa-val-unsupervised-cls-v3-top5 8 &> ./logs/notrain/vqa-slidevqa-val-unsupervised-cls-v5-top3.log
# wait

# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-slidevqa-val-unsupervised-cls-v3-top3-test2 8 &> ./logs/notrain/vqa-slidevqa-val-unsupervised-cls-v3-top3-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-slidevqa-val-unsupervised-cls-v3-top5-test2 8 &> ./logs/notrain/vqa-slidevqa-val-unsupervised-cls-v3-top5-test2.log
# wait

# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-slidevqa-val-unsupervised-cls-v4-top3_mix 8 &> ./logs/notrain/vqa-slidevqa-val-unsupervised-cls-v4-top3_mix.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-slidevqa-val-unsupervised-cls-v4-top5_mix 8 &> ./logs/notrain/vqa-slidevqa-val-unsupervised-cls-v4-top5_mix.log
# wait

# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-slidevqa-val-unsupervised-cls-v4-top3_mix-test2 8 &> ./logs/notrain/vqa-slidevqa-val-unsupervised-cls-v4-top3_mix-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/huoluo.wx/models/Qwen__Qwen2-VL-7B-Instruct vqa-slidevqa-val-unsupervised-cls-v4-top5_mix-test2 8 &> ./logs/notrain/vqa-slidevqa-val-unsupervised-cls-v4-top5_mix-test2.log
# wait

# # sft

# ## mpdocvqa
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-mpdocvqa-val 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-mpdocvqa-val.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-mpdocvqa-val-unsupervised-cls-v3-top3 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-mpdocvqa-val-unsupervised-cls-v3-top3.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-mpdocvqa-val-unsupervised-cls-v3-top5 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-mpdocvqa-val-unsupervised-cls-v3-top5.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/qa-mpdocvqa-val-unsupervised-cls-v4-top3_mix.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix.log
# wait

# ## dude
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-dude-mpdocvqa-val 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-dude-mpdocvqa-val.log
# wait
# CUDA_VISIBLE_DEVICES=4 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3.log &
# wait
# CUDA_VISIBLE_DEVICES=5 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5.log &
# wait
# CUDA_VISIBLE_DEVICES=6 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix.log &
# wait
# CUDA_VISIBLE_DEVICES=7 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix.log &
# wait

# ## newsvqa
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-newsvqa-val 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-newsvqa-val.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-newsvqa-val-unsupervised-cls-v3-top3 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-newsvqa-val-unsupervised-cls-v3-top3.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-newsvqa-val-unsupervised-cls-v3-top5 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-newsvqa-val-unsupervised-cls-v3-top5.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-newsvqa-val-unsupervised-cls-v4-top3_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-newsvqa-val-unsupervised-cls-v4-top3_mix.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-newsvqa-val-unsupervised-cls-v4-top5_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-newsvqa-val-unsupervised-cls-v4-top5_mix.log
# wait

# ## slidevqa
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-slidevqa-val 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-slidevqa-val.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-newsvqa-val-unsupervised-cls-v3-top3 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-newsvqa-val-unsupervised-cls-v3-top3.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-slidevqa-val-unsupervised-cls-v3-top3 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-slidevqa-val-unsupervised-cls-v3-top3.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-slidevqa-val-unsupervised-cls-v4-top3_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/vqa-slidevqa-val-unsupervised-cls-v4-top3_mix.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/mpdocvqa_dude_newsvqa_slidevqa/1e-6/qwen2-vl-7b-instruct/v11-20241212-001249/checkpoint-18868/ vqa-slidevqa-val-unsupervised-cls-v4-top5_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_sft_100352/qa-slidevqa-val-unsupervised-cls-v4-top5_mix.log
# wait

# sft 200704

## mpdocvqa
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-mpdocvqa-val 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-mpdocvqa-val.log
# wait
# CUDA_VISIBLE_DEVICES=0 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-mpdocvqa-val-unsupervised-cls-v3-top3 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-mpdocvqa-val-unsupervised-cls-v3-top3.log &
# CUDA_VISIBLE_DEVICES=1 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-mpdocvqa-val-unsupervised-cls-v3-top5 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-mpdocvqa-val-unsupervised-cls-v3-top5.log &
# CUDA_VISIBLE_DEVICES=2 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix.log &
# CUDA_VISIBLE_DEVICES=3 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix.log &

# CUDA_VISIBLE_DEVICES=4 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-mpdocvqa-val-unsupervised-cls-v3-top3-test2 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-mpdocvqa-val-unsupervised-cls-v3-top3-test2.log &
# CUDA_VISIBLE_DEVICES=5 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-mpdocvqa-val-unsupervised-cls-v3-top5-test2 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-mpdocvqa-val-unsupervised-cls-v3-top5-test2.log &
# CUDA_VISIBLE_DEVICES=6 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix-test2 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-mpdocvqa-val-unsupervised-cls-v4-top3_mix-test2.log &
# CUDA_VISIBLE_DEVICES=7 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix-test2 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-mpdocvqa-val-unsupervised-cls-v4-top5_mix-test2.log &

# wait
## dude

# CUDA_VISIBLE_DEVICES=0 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3.log &
# # wait
# CUDA_VISIBLE_DEVICES=1 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5.log &
# # wait
# CUDA_VISIBLE_DEVICES=2 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix.log &
# # wait
# CUDA_VISIBLE_DEVICES=3 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix.log 


# CUDA_VISIBLE_DEVICES=4 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3-test2 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top3-test2.log &
# # wait
# CUDA_VISIBLE_DEVICES=5 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5-test2 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-dude-mpdocvqa-val-unsupervised-cls-v3-top5-test2.log &
# # wait
# CUDA_VISIBLE_DEVICES=6 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix-test2 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top3_mix-test2.log &
# # wait
# CUDA_VISIBLE_DEVICES=7 bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix-test2 1 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-dude-mpdocvqa-val-unsupervised-cls-v4-top5_mix-test2.log 
# wait

## newsvqa
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val-unsupervised-cls-v3-top3 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val-unsupervised-cls-v3-top3.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val-unsupervised-cls-v3-top5 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val-unsupervised-cls-v3-top5.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val-unsupervised-cls-v4-top3_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val-unsupervised-cls-v4-top3_mix.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val-unsupervised-cls-v4-top5_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val-unsupervised-cls-v4-top5_mix.log
# wait

# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val-unsupervised-cls-v3-top3-test2 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val-unsupervised-cls-v3-top3-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val-unsupervised-cls-v3-top5-test2 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val-unsupervised-cls-v3-top5-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val-unsupervised-cls-v4-top3_mix-test2 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val-unsupervised-cls-v4-top3_mix-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val-unsupervised-cls-v4-top5_mix-test2 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val-unsupervised-cls-v4-top5_mix-test2.log
# wait

# slidevqa
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-slidevqa-val 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-slidevqa-val.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val-unsupervised-cls-v3-top3 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val-unsupervised-cls-v3-top3.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-slidevqa-val-unsupervised-cls-v3-top5 7 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-slidevqa-val-unsupervised-cls-v3-top5.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-slidevqa-val-unsupervised-cls-v4-top3_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-slidevqa-val-unsupervised-cls-v4-top3_mix.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-slidevqa-val-unsupervised-cls-v4-top5_mix 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-slidevqa-val-unsupervised-cls-v4-top5_mix.log
# wait

# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-newsvqa-val-unsupervised-cls-v3-top3-test2 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-newsvqa-val-unsupervised-cls-v3-top3-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-slidevqa-val-unsupervised-cls-v3-top5-test2 7 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-slidevqa-val-unsupervised-cls-v3-top5-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-slidevqa-val-unsupervised-cls-v4-top3_mix-test2 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-slidevqa-val-unsupervised-cls-v4-top3_mix-test2.log
# wait
# bash ./mpdoc_vqa_infer_mpdocvqa_dude_newsvqa_slidevqa.sh /gruntdata/heyuan67/zqp/qp_lm_models/llamafactory/mpdocvqa_dude_newsvqa_slidevqa_448/full/sft_all/ vqa-slidevqa-val-unsupervised-cls-v4-top5_mix-test2 8 &> ./logs/mpdocvqa_dude_newsvqa_slidevqa_200704/vqa-slidevqa-val-unsupervised-cls-v4-top5_mix-test2.log
# wait

bash run_mpdocvqa_1e-6.sh



