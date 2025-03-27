import argparse
import itertools
import json
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional

import torch
from model import Qwen2VLForConditionalGeneration_Unsupervised_CLS2
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from copy import deepcopy

ds_collections = {
    'mpdocvqa_val': {
        'train': None,
        'test': '/mntnlp/qp_mm_data/docvqa/mpdoc/mpdoc_val_swift_qwen2vl.jsonl',
        'annotation': '/mntnlp/qp_mm_data/docvqa/mpdoc/val.json',
        'annotation_details': [f'/mntnlp/qp_mm_data/docvqa/mpdoc/val_{idx}.json' for idx in range(5)],
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'newsvqa_val': {
        'train': None,
        'test': '/gruntdata/heyuan67/zqp/NewsVQA/ICDAR2023/swift_val.jsonl',
        'annotation': '/gruntdata/heyuan67/zqp/NewsVQA/ICDAR2023/annotation/swift_newsvideoqa_version_2_val_release2.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'slidevqa_val': {
        'train': None,
        'test': '/gruntdata/heyuan67/zqp/SlideVQA/swift_val_slidevqa.jsonl',
        'annotation': '/gruntdata/heyuan67/zqp/SlideVQA/internvl_val_slidevqa_annotations.json',
        'metric': 'anls',
        'max_new_tokens': 100
    },
    'dude_mpdocvqa_val': {
        'train': None,
        'test': '/gruntdata/heyuan67/zqp/DUDE_loader/internvl_val_dude_swift_mpdocqa.jsonl',
        'annotation': '/gruntdata/heyuan67/zqp/DUDE_loader/internvl_val_dude_mpdocqa_annotations.json',
        'annotation_details': [f'/gruntdata/heyuan67/zqp/DUDE_loader/internvl_val_dude_mpdocqa_annotations_{idx}.json' for idx in range(5)],
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'mpdocvqa_train1000': {
        'train': None,
        'test': '/mntnlp/qp_mm_data/docvqa/mpdoc/mpdoc_train_swift_qwen2vl_sample1000.jsonl',
        'annotation': '/mntnlp/qp_mm_data/docvqa/mpdoc/mpdoc_train_ann_sample1000.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    }
}

def collate_fn(batches):
    inputs = [_['inputs'] for _ in batches]
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    num_token_list = [_['num_token_list'] for _ in batches]
    messages_high = [_['messages_high'] for _ in batches]
    messages_low = [_['messages_low'] for _ in batches]
    answer_page_idxs = [_['answer_page_idx'] for _ in batches]

    return inputs, questions, question_ids, annotations, num_token_list, messages_high, messages_low, answer_page_idxs

system_message = "You are a helpful assistant."

class VQADataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, image_process_args, processor):
        self.test = open(test).readlines()
        self.prompt = prompt
        self.image_process_args = image_process_args
        self.processor = processor
        self.dynamic = self.image_process_args['dynamic']

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image_paths, query, question_id, annotation, answer_page_idx = data['images'], data[
            'query'].replace('<image>',''), data['id'], data.get('answers', None), data['answer_page_idx']

        max_pixels_high = self.image_process_args['max_pixels_high']
        max_pixels_low = self.image_process_args['max_pixels_low']

        # max_pixels_high = self.image_process_args['max_pixels_all']
        # if max_pixels_high < self.image_process_args['max_pixels_low']:
        #     max_pixels_high = self.image_process_args['max_pixels_low']

        messages_high = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image_path, "max_pixels": max_pixels_high} for image_path in image_paths] + [{"type": "text", "text": self.prompt+query}]
            },
        ]

        messages_low = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image_path, "max_pixels": max_pixels_low} for image_path in image_paths] + [{"type": "text", "text": self.prompt+query}]
            },
        ]
        # print(messages_low)
        image_inputs_low, video_inputs_low = process_vision_info(messages_low)

        text_low = processor.tokenizer.apply_chat_template(
            messages_low, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

        inputs_low = processor(
            text=[text_low],
            images=image_inputs_low,
            videos=video_inputs_low,
            padding=True,
            return_tensors="pt",
        )

        vision_pos_pad = torch.where(inputs_low['input_ids'] == 151655)[1]
        num_token_list = []
        cnt = 1
        for idx in range(1,len(vision_pos_pad)):
            if vision_pos_pad[idx] - vision_pos_pad[idx-1] > 1:
                num_token_list.append(cnt)
                cnt = 0
            else:
                cnt += 1
        num_token_list.append(cnt)

        return {
            'question_id': question_id,
            'question': query,
            'inputs': inputs_low,
            'annotation': annotation,
            'num_token_list': num_token_list,
            'messages_high': messages_high,
            'messages_low': messages_low,
            'answer_page_idx': answer_page_idx
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(response):
    response = response.strip().split('.')[0].split(
        ',')[0].split('!')[0].lower()
    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]
    response = response.strip()
    return response

def predict_page(
        inputs,
        qwen2vl_generation_model,
        max_new_tokens=1024,
        num_token_list=None,
        attn_pool='max',
        cls_num_layers=1,
        select_mode='topk',
        select_image_num=3,
        **kwargs):

    inputs['cache_position'] = torch.arange(len(inputs.input_ids[0]), device=inputs.input_ids.device)
    inputs['cache'] = True
    new_inputs = qwen2vl_generation_model.prepare_inputs_for_generation(**inputs, max_new_tokens=max_new_tokens)
    input_ids = new_inputs['input_ids']
    image_grid_thw = new_inputs['image_grid_thw']
    video_grid_thw = None
    rope_deltas = new_inputs['rope_deltas']
    attention_mask = new_inputs['attention_mask']
    pixel_values = new_inputs['pixel_values']
    position_ids = new_inputs['position_ids']

    inputs_embeds = qwen2vl_generation_model.model.embed_tokens(input_ids)

    pixel_values = pixel_values.type(qwen2vl_generation_model.visual.get_dtype())
    image_embeds = qwen2vl_generation_model.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
    image_mask = input_ids == qwen2vl_generation_model.config.image_token_id

    inputs_embeds[image_mask] = image_embeds
    attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None and input_ids is not None:
        position_ids, _ = qwen2vl_generation_model.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)

    selected = image_mask
    cache_position = None
    past_key_values = None
    output_attentions = True

    causal_mask = qwen2vl_generation_model.model._update_causal_mask(
    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds
    # create position embeddings to be shared across the decoder layers
    position_embeddings = qwen2vl_generation_model.model.rotary_emb(hidden_states, position_ids.clone())

    next_decoder_cache = None

    for layer_idx in range(cls_num_layers):
        if layer_idx == cls_num_layers-1:
            tmp_layer_outputs = qwen2vl_generation_model.model.layers[layer_idx](
                hidden_states=hidden_states,
                position_ids=position_ids.clone(),
                attention_mask=causal_mask,
                past_key_value=past_key_values,
                use_cache=False,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                attn_pool=attn_pool,
                selected=selected,
                num_token_list=num_token_list,
            )
        else:
            tmp_layer_outputs = qwen2vl_generation_model.model.layers[layer_idx](
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=causal_mask,
                past_key_value=past_key_values,
                use_cache=False,
                output_attentions=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = tmp_layer_outputs[0]
    attentions = tmp_layer_outputs[1][0]

    if select_mode == 'dynamic':
        page_pred = torch.where(attentions != 0)[0]
    else:
        page_pred = torch.topk(attentions, min(select_image_num, len(attentions)), dim=-1).indices

    return page_pred

def get_new_input(page_pred,
                  messages_high=None,
                  messages_low=None,
                  processor=None,
                  device=None,
                  use_all=True):
    new_messages = [
    {
        "role": "user",
        "content": []
    }]
    new_images = []

    select_num = len(page_pred)

    images_high, images_low = messages_high['content'][:-1], messages_low['content'][:-1]  
    high, low = images_high[0]['max_pixels'], images_low[0]['max_pixels']
    max_pixels_high = high // select_num
    if max_pixels_high < low:
        max_pixels_high = low

    for idx in range(len(images_high)):
        images_high[idx]['max_pixels'] = max_pixels_high

    for idx in range(len(images_low)):
        if idx in page_pred:
            new_images.append(images_high[idx])
        elif use_all:
            new_images.append(images_low[idx])

    new_messages[0]['content'] =  new_images + messages_low['content'][-1:]
    new_image_inputs, new_video_inputs = process_vision_info(new_messages)
    new_text = processor.tokenizer.apply_chat_template(
        new_messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)

    new_inputs = processor(
        text=[new_text],
        images=new_image_inputs,
        videos=new_video_inputs,
        padding=True,
        return_tensors="pt",).to(device)
    return new_inputs

def evaluate_chat_model(model, processor, image_process_args):
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    # infovqa_prompt = 'Answer the question directly.'
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    dude_prompt = 'Answer the question using a single word or phrase or sentence.'
    ai2d_prompt = ''
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'ai2d' in ds_name:
            input_prompt = ai2d_prompt
        elif 'infographicsvqa' in ds_name:
            input_prompt = infovqa_prompt
        elif 'dude' in ds_name:
            input_prompt = dude_prompt
        else:
            input_prompt = base_prompt

        dataset = VQADataset(
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
            image_process_args=image_process_args, 
            processor=processor
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        outputs = []
        pred_acc = []
        outputs_details = {0:[], 1:[], 2:[], 3:[], 4:[]}

        for _, (inputs, questions, question_ids, annotations, num_token_list, messages_high, messages_low, answer_page_idxs) in tqdm(enumerate(dataloader)):
            # get page pred
            page_pred = predict_page(inputs[0].to(model.device),
                        qwen2vl_generation_model=model,
                        max_new_tokens=1024,
                        num_token_list=num_token_list,
                        attn_pool='max',
                        cls_num_layers=1,
                        select_mode='topk')

            if type(answer_page_idxs[0]) != list:
                pred_acc.append(answer_page_idxs[0] in page_pred)
            else:
                if -1 not in answer_page_idxs[0] and max(answer_page_idxs[0]) < len(messages_high[0][0]):
                    r = True
                    for answer in answer_page_idxs[0]:
                        r &= (answer in page_pred)
                    pred_acc.append(r)

            inputs = get_new_input(page_pred=page_pred,
                                    messages_high=messages_high[0][0],
                                    messages_low=messages_low[0][0],
                                    processor=processor,
                                    device=model.device)

            generated_ids = model.generate(**inputs.to(model.device),
                                           max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            pred = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            answers = pred

            for question, question_id, answer, annotation in zip(questions, question_ids, answers, annotations):
                if ds_name in ['dude_mpdocvqa_val', 'mpdocvqa_val', 'mpdocvqa_train1000', 'docvqa_val', 
                                 'mpdocvqa_2stage_val', 'dude_mpdocvqa_2stage_val', 'newsvqa_val', 'slidevqa_val']:
                    outputs.append({
                        'question': question,
                        'questionId': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                    key_index = len(inputs[0].pixel_values) // 5
                    if key_index > 4:
                        key_index = 4
                    outputs_details[key_index].append({
                        'num_image': len(inputs[0].pixel_values),
                        'question': question,
                        'questionId': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                else:
                    raise NotImplementedError

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs_details = {0:[None for _ in range(world_size)], 1:[None for _ in range(world_size)], 2:[None for _ in range(world_size)], 3:[None for _ in range(world_size)], 4:[None for _ in range(world_size)]}

        for key_index in range(5):
            torch.distributed.all_gather_object(merged_outputs_details[key_index], json.dumps(outputs_details[key_index]))

        for key_index in range(5):
            merged_outputs_details_tmp = [json.loads(_) for _ in merged_outputs_details[key_index]]
            merged_outputs_details[key_index] = [_ for _ in itertools.chain.from_iterable(merged_outputs_details_tmp)]

        page_predict_acc_all = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(page_predict_acc_all, pred_acc)

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            page_predict_acc_all = [pre for pres in page_predict_acc_all for pre in pres]
            pred_acc = torch.tensor(page_predict_acc_all).float().mean(-1)
            print(pred_acc)

            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

            results_file_indexs = []
            for key_index in range(5):
                results_file_index = f'{ds_name}_{time_prefix}_{key_index}.json'
                results_file_index = os.path.join(args.out_dir, results_file_index)
                json.dump(merged_outputs_details[key_index], open(results_file_index, 'w'))
                results_file_indexs.append(results_file_index)

            if ds_collections[ds_name]['metric'] == 'anls':
                with open(results_file, mode='w', encoding='utf-8') as f:
                    json.dump(merged_outputs, f, ensure_ascii=False)
                print('python utils/infographicsvqa_eval.py -g ' +
                      ds_collections[ds_name]['annotation'] + ' -s ' +
                      results_file)
                os.system('python utils/infographicsvqa_eval.py -g ' +
                          ds_collections[ds_name]['annotation'] + ' -s ' +
                          results_file)

                if ds_collections[ds_name].get('annotation_details',None) is not None:
                    for i in range(5):
                        with open(results_file, mode='w', encoding='utf-8') as f:
                            json.dump(merged_outputs, f, ensure_ascii=False)
                        print('python utils/infographicsvqa_eval.py -g ' +
                            ds_collections[ds_name]['annotation_details'][i] + ' -s ' +
                            results_file_indexs[i])
                        os.system('python utils/infographicsvqa_eval.py -g ' +
                            ds_collections[ds_name]['annotation_details'][i] + ' -s ' +
                            results_file_indexs[i])

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-pixels-all', type=float, default=1003520)
    parser.add_argument('--max-pixels', type=float, default=100352)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model = Qwen2VLForConditionalGeneration_Unsupervised_CLS2.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            # device_map='auto',
            ).cuda()
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    image_process_args = {
        "max_pixels_low": args.max_pixels,
        "max_pixels_high": args.max_pixels * 5,
        "max_pixels_all": args.max_pixels_all,
        "dynamic": args.dynamic
        # "max_pixels": 501760
        # "resized_height": 400,
        # "resized_width": 300
    }
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    evaluate_chat_model(model, processor, image_process_args)
