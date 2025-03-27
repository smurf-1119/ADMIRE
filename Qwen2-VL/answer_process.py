import json

with open("/mntnlp/zqp/physical_report/Qwen2-VL/infer_res_for_high_fps/test_for_high_fps_merged.json", "r") as f:
    prediction = json.load(f)

def get_label(answer_id):
    answer_id = answer_id[0]
    if answer_id == "A":
        return 0
    elif answer_id == "B":
        return 1
    elif answer_id == "C":
        return 2
    else:
        print("No such answer_id")

processed_prediction = {}
for video in prediction:
    if video == "meta_info":
        # prediction.pop(video)
        continue
    processed_prediction[video.replace(".mp4", "")] = []

    for question in prediction[video]:
        new_res  = {}
        new_res["answer_id"] = get_label(question["answer_id"])
        new_res['answer'] = question['answer_text'][new_res["answer_id"]]
        new_res['id'] = question['id']
        processed_prediction[video.replace(".mp4", "")].append(new_res)

question["answer_id"]

with open('/mntnlp/zqp/physical_report/Qwen2-VL/infer_res_for_high_fps/test_for_high_fps_merged_processed.json', 'w')as f:
    json.dump(processed_prediction, f)


