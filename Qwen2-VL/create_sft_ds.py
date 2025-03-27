import json
import os
from sklearn.model_selection import KFold

root_dir = "./data"
query_template = """Answer the following question based on the provided video.
<video>
Question: {question}

Options:
A. {option[0]}
B. {option[1]}
C. {option[2]}

Your answer (choose one of the options): """
answer_map = {
    0: "A",
    1: "B",
    2: "C",
}
answer_template = """{answer_id}. {answer_text}"""

save_path = "./data/sft_dataset_cv"
if not os.path.exists(save_path):
    os.makedirs(save_path)

def write_to_jsonl(data, file_path):
    with open(file_path, "w") as save_file:
        for item in data:
            save_file.write(json.dumps(item) + "\n")
def create_sft_ques_dict(question, video_id, root_dir):
    if question.get("answer_id") is None:
        return {
            "system": "You are a helpful assistant. You are good at answering questions about the video. You should think step by step.",
            "query": query_template.format(
                question=question["question"], option=question["options"]
            ),
            "videos": [os.path.join(root_dir, "videos", video_id+".mp4")],
            "history": [],
            "video_id": video_id,
            "id": question["id"],
            "options": question["options"],
        }
    else:
        return {
            "query": query_template.format(
                question=question["question"], option=question["options"]
            ),
            "system": "You are a helpful assistant. You are good at answering questions about the video. You should think step by step.",
            "response": answer_template.format(
                answer_id=answer_map[question["answer_id"]],
                answer_text=question["options"][question["answer_id"]],
            ),
            "history": [],
            "videos": [os.path.join(root_dir, "videos", video_id+".mp4")],
            "area": question["area"],
            "reasoning": question["reasoning"],
            "tag": question["tag"],
        }
# 处理训练数据
train_data = []
data_path = os.path.join(root_dir, "mc_question_train.json")
with open(data_path, "r") as f:
    data = json.load(f)

for video_id, video_data in data.items():
    questions = video_data["mc_question"]
    for question in questions:
        sft_ques_dict = create_sft_ques_dict(question, video_id, root_dir)
        train_data.append(sft_ques_dict)

# 创建4个分割
kf = KFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
    fold_train_data = [train_data[i] for i in train_index]
    fold_val_data = [train_data[i] for i in val_index]
    
    write_to_jsonl(fold_train_data, os.path.join(save_path, f"train_fold{fold}.jsonl"))
    write_to_jsonl(fold_val_data, os.path.join(save_path, f"val_fold{fold}.jsonl"))

# 处理完整的训练数据
write_to_jsonl(train_data, os.path.join(save_path, "train.jsonl"))

# 处理验证和测试数据
for split in ["test"]:
    data_path = os.path.join(root_dir, f"mc_question_{split}.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    
    split_data = []
    for video_id, video_data in data.items():
        questions = video_data["mc_question"]
        for question in questions:
            sft_ques_dict = create_sft_ques_dict(question, video_id, root_dir)
            split_data.append(sft_ques_dict)
    
    write_to_jsonl(split_data, os.path.join(save_path, f"{split}.jsonl"))

print("所有JSONL文件已在sft_dataset目录中创建完成。")


