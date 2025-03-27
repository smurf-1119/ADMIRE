import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "model_weights/llamafactory/sft_all_unfreezevit/checkpoint-5000", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "model_weights/llamafactory/sft_all_unfreezevit/checkpoint-5000",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("model_weights/llamafactory/sft_all_unfreezevit/checkpoint-5000")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

data = [json.loads(line) for line in open('/mntnlp/qp_mm_data/docvqa/mpdoc/mpdoc_val_swift_qwen2vl.jsonl')]
image_process_args = {
        "max_pixels": 200704,
        # "resized_height": 400,
        # "resized_width": 300
    }
prompt = 'Answer the question using a single word or phrase. '
for line in tqdm(data):
    image_paths, query, question_id, annotation = line['images'], line[
            'query'], line['id'], line.get('answers', None)

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image_path, **image_process_args} for image_path in image_paths] + [{"type": "text", "text": prompt+query.replace('<image>','')}]
        }
    ]
    print(messages)
    # Preparation for inference
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text,annotation)