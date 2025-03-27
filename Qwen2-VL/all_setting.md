
# 环境安装
py310
torch2.4
```
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .

# Qwen2-VL-7B-Instruct 依赖
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
pip install qwen-vl-utils
```
pip install markupsafe==2.0.0
pip install ./transformers
pip uninstall accelerate
pip install accelerate
# 数据准备

data 目录底下包含三个文件：

1. videos 【需要下载train和test】
2. mc_queston_train.json
3. mc_queston_test.json

先构建数据集：
```
python create_sft_ds.py
```

data目录底下会生成一个sft_dataset_cv目录，包含：

1. train.jsonl
2. test.jsonl
3. valid.jsonl
4. train_fold0.jsonl
5. train_fold1.jsonl
6. train_fold2.jsonl
7. train_fold3.jsonl
8. val_fold0.jsonl
9. val_fold1.jsonl
10. val_fold2.jsonl
11. val_fold3.jsonl

我们可以用fold也可以不用，看情况。


然后就可以跑sft了，一些常见的参数可以自己调一下
```bash
FPS=2 MAX_PIXELS=1003520 CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
  --model_type qwen2-vl-7b-instruct \
  --model_id_or_path /home/share/pyz/model_weight/Qwen2-VL-7B-Instruct \
  --sft_type lora \
  --dataset /home/pyz/code/long_video_qa/m-QA/data/sft_dataset/train.jsonl \
  --deepspeed default-zero2 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules AUTO \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 4 \
  --freeze_vit False \
  --eval_steps 200 \
  --logging_steps 5 \
  --batch_size 2 \
  --dataloader_num_workers 1 \
  --num_train_epochs 2 \
```
跑完后，需要merge lora权重
```bash
CUDA_VISIBLE_DEVICES=0 swift export --ckpt_dir 'xxx/checkpoint-93' --merge_lora true
```

## Infer
merge完后，在output底下有一个checkpoint-xxx-merged的文件夹，里面就是何必的权重然后就可以跑infer了

**跑之前修改一下路径**

```
# 配置变量 手动改一下脚本，没有从环境变量读取
MODEL_PATH = "/home/share/pyz/model_weight/Qwen2-VL-7B-Instruct"
DEVICE = "cuda:1"
TEST_DATA_PATH = "./data/mc_question_test.json"
VIDEO_DIR = "./data/videos"
OUTPUT_DIR = "./infer_res"
OUTPUT_FILE = "test_baseline_batch.json"
TOTAL_NUM = 11528
```
```python 
python infer_qwenvl2.py
```


## 后处理
```
python answer_process.py

```

提交 `./test_baseline_processed.json`
