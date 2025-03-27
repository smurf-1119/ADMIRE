import json
import os
import torch
import torch.multiprocessing as mp
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from collections import defaultdict
import argparse

# 配置变量
MODEL_PATH = "/mntnlp/zqp/physical_report/Qwen2-VL/output/qwen2-vl-7b-instruct/v25-20240912-002220/checkpoint-456-merged"
VIDEO_DIR = "./data/videos"
OUTPUT_DIR = "./infer_res_for_high_fps"

video_process_args = {
    "nframes": 60,
    "max_pixels": 151200,
    "resized_height": 360,
    "resized_width": 420,
}

query_template = """Answer the following question based on the provided video.
<video>
Question: {question}

Options:
A. {option[0]}
B. {option[1]}
C. {option[2]}

Your answer (choose one of the options): """

def process_group(rank, world_size, args):
    torch.cuda.set_device(rank)
    
    # 加载模型和处理器
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map=f"cuda:{rank}",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    group_id = rank + 1  # 每个rank处理一个组
    
    # 加载组数据
    with open(f"./data/grouped_data/group_{group_id}.json", "r") as f:
        data = json.load(f)
    
    total_res = defaultdict(list)
    for video_id, video_data in tqdm(data.items(), desc=f"Group {group_id} on GPU {rank}"):
        video_path = f"{VIDEO_DIR}/{video_id}.mp4"
        video_message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, **video_process_args}
                ],
            }
        ]
        image_inputs, video_inputs = process_vision_info(video_message)
        
        for question_item in video_data["mc_question"]:
            question = question_item["question"]
            options = question_item["options"]
            query = query_template.format(question=question, option=options)
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You are good at answering questions about the video. You should think step by step."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer the following question based on the provided video.\n"},
                        {"type": "video", "video": video_path, **video_process_args},
                        {"type": "text", "text": query},
                    ],
                },
            ]
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            responses = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            total_res[os.path.basename(video_path)].append(
                {
                    "id": question_item["id"],
                    "answer_id": responses[0],
                    "answer_text": question_item["options"],
                }
            )
    
            # 保存结果
            output_file = f"{OUTPUT_DIR}/test_for_high_fps_group_{group_id}.json"
            with open(output_file, "w") as f:
                json.dump(total_res, f)

def merge_results():
    merged_results = {}
    for i in range(8):  # 合并8个组的结果
        file = f"{OUTPUT_DIR}/test_for_high_fps_group_{i + 1}.json"
        with open(file, 'r') as f:
            data = json.load(f)
            merged_results.update(data)
    
    output_file = f"{OUTPUT_DIR}/test_for_high_fps_merged.json"
    with open(output_file, 'w') as f:
        json.dump(merged_results, f, indent=2)
    
    print(f"合并结果已保存到: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    mp.spawn(process_group, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True)

    # 合并结果
    final_output = merge_results()
    print(f"最终合并结果已保存到: {final_output}")

    # 可选：删除中间结果文件
    for i in range(8):
        os.remove(f"{OUTPUT_DIR}/test_baseline_fold3_group_{i + 1}.json")
    print("中间结果文件已删除")

if __name__ == "__main__":
    main()
