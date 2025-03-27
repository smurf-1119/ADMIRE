import json
import os
from collections import defaultdict

TEST_DATA_PATH = "./data/mc_question_test.json"
with open(TEST_DATA_PATH, "r") as f:
    data = json.load(f)

# 获取视频长度信息
video_lengths = {}
for video_id, video_data in data.items():
    num_frames = video_data['metadata']['num_frames']
    frame_rate = video_data['metadata']['frame_rate']
    video_length = num_frames / frame_rate
    video_lengths[video_id] = video_length

# 定义分组数量
k = 8

# 计算目标平均长度
total_length = sum(video_lengths.values())
target_avg_length = total_length / k

# 创建分组
groups = defaultdict(list)
group_lengths = defaultdict(float)

sorted_videos = sorted(video_lengths.items(), key=lambda x: x[1], reverse=True)

for video_id, length in sorted_videos:
    # 找到当前长度最短的组
    current_group = min(range(k), key=lambda i: group_lengths[i])
    
    groups[current_group].append(video_id)
    group_lengths[current_group] += length

# 打印分组结果
print(f"视频已被分成 {k} 组:")
for group_id, video_ids in groups.items():
    avg_length = group_lengths[group_id] / len(video_ids)
    print(f"组 {group_id + 1}: {len(video_ids)} 个视频")
    print(f"  平均长度: {avg_length:.2f} 秒")

# 将分组结果保存到文件
output_dir = "./data/grouped_data"
os.makedirs(output_dir, exist_ok=True)

for group_id, video_ids in groups.items():
    group_data = {video_id: data[video_id] for video_id in video_ids}
    output_file = os.path.join(output_dir, f"group_{group_id + 1}.json")
    with open(output_file, "w") as f:
        json.dump(group_data, f, ensure_ascii=False, indent=2)

print(f"分组数据已保存到 {output_dir} 目录")