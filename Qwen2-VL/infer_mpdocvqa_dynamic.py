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
from model import Qwen2VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

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
    'dude_mpdocvqa_val': {
        'train': None,
        'test': '/gruntdata/heyuan67/zqp/DUDE_loader/internvl_val_dude_swift_mpdocqa.jsonl',
        'annotation': '/gruntdata/heyuan67/zqp/DUDE_loader/internvl_val_dude_mpdocqa_annotations.json',
        'annotation_details': [f'/gruntdata/heyuan67/zqp/DUDE_loader/internvl_val_dude_mpdocqa_annotations_{idx}.json' for idx in range(5)],
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
    num_image_list = [_['num_image'] for _ in batches]
    avg_vtokens_list = [_['avg_vtokens'] for _ in batches]
    total_tokens_list = [_['total_tokens'] for _ in batches]

    return inputs, questions, question_ids, annotations, num_image_list, avg_vtokens_list, total_tokens_list

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
        image_paths, query, question_id, annotation = data['images'], data[
            'query'].replace('<image>',''), data['id'], data.get('answers', None)

        if self.dynamic:
            max_pixels = self.image_process_args['max_pixels_all'] // len(data['images'])
            if max_pixels < self.image_process_args['max_pixels']:
                max_pixels = self.image_process_args['max_pixels']
            image_process_args_tmp = {"max_pixels":max_pixels}
        else:
            image_process_args_tmp = {"max_pixels":self.image_process_args['max_pixels']}
            # image_process_args_tmp = {"max_resolution":448}
        # print(image_process_args_tmp)

        # image_inputs, video_inputs = process_vision_info(image_message)
        messages = [
            # {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [{"type": "image", "image": image_path, **image_process_args_tmp} for image_path in image_paths] + [{"type": "text", "text": self.prompt+query}]
            },
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        avg_vtokens = sum(inputs['input_ids'][0]==151655) // len(data['images'])
        total_tokens = len(inputs['input_ids'][0])

        return {
            'question_id': question_id,
            'question': query,
            'inputs': inputs,
            'annotation': annotation,
            'num_image': len(image_paths),
            'avg_vtokens': avg_vtokens,
            'total_tokens': total_tokens
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
        outputs_details = {0:[], 1:[], 2:[], 3:[], 4:[]}
        avg_vtokens = []
        total_tokens = []
        for _, (inputs, questions, question_ids, annotations, num_image_list, avg_vtokens_list, total_tokens_list) in tqdm(enumerate(dataloader)):
            generated_ids = model.generate(**inputs[0].to(model.device), max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs[0].input_ids, generated_ids)
            ]
            pred = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answers = pred
            avg_vtokens.append(avg_vtokens_list[0])
            total_tokens.append(total_tokens_list[0])

            for question, question_id, answer, annotation in zip(questions, question_ids, answers, annotations):
                if ds_name in ['dude_mpdocvqa_val', 'mpdocvqa_val', 'mpdocvqa_train1000', 'docvqa_val', 
                                 'mpdocvqa_2stage_val', 'dude_mpdocvqa_2stage_val', 'newsvqa_val', 'slidevqa_val']:
                    outputs.append({
                        'question': question,
                        'questionId': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })

                    key_index = num_image_list[0] // 5
                    if key_index > 4:
                        key_index = 4
                    outputs_details[key_index].append({
                        'num_image': num_image_list[0],
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

        merged_avg_vtokens = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_avg_vtokens, avg_vtokens)

        merged_total_tokens = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_total_tokens, total_tokens)

        merged_outputs_details = {0:[None for _ in range(world_size)], 1:[None for _ in range(world_size)], 2:[None for _ in range(world_size)], 3:[None for _ in range(world_size)], 4:[None for _ in range(world_size)]}

        for key_index in range(5):
            torch.distributed.all_gather_object(merged_outputs_details[key_index], json.dumps(outputs_details[key_index]))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        merged_avg_vtokens = [_ for _ in itertools.chain.from_iterable(merged_avg_vtokens)]
        merged_total_tokens = [_ for _ in itertools.chain.from_iterable(merged_total_tokens)]

        for key_index in range(5):
            merged_outputs_details_tmp = [json.loads(_) for _ in merged_outputs_details[key_index]]
            merged_outputs_details[key_index] = [_ for _ in itertools.chain.from_iterable(merged_outputs_details_tmp)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

            print('avg vtokens', sum(merged_avg_vtokens) / len(merged_avg_vtokens))
            print('avg total tokens', sum(merged_total_tokens) / len(merged_total_tokens))

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
    parser.add_argument('--max-pixels', type=float, default=200704)
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

    model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            # device_map='auto',
            ).cuda()
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    image_process_args = {
        "max_pixels": args.max_pixels,
        "max_pixels_all": args.max_pixels_all,
        "dynamic": args.dynamic
        # "max_pixels": 501760
        # "resized_height": 400,
        # "resized_width": 300
    }
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    evaluate_chat_model(model, processor, image_process_args)
