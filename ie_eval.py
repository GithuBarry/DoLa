# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/alibaba/FederatedScope/blob/dev/llm/federatedscope/llm/eval/eval_for_gsm8k/eval.py

import argparse
import json
import os
import random
import re
import ssl
import urllib.request
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

import transformers
from dola import DoLa

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = True


def load_jsonl(file_path,
               instruction='instruction',
               input='input',
               is_gzip=False):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None)
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())
    return path


def build_prompt(input_text):
    return (f"<s>[INST] <<SYS>> You are a helpful assistant. Output in JSON only.<</SYS>> original text: {input_text}\n"
            "Given above text, there might be zero or more terrorist events in it. "
            "If to fill [{\"incident_type\": \"attack\", \"PerpInd\": [...], \"PerpOrg\": [...], \"Target\": [...], \"Victim\": [...], \"Weapon\": [...]}] "
            "for every event actual present, it would be [/INST]")


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    # Regular expression pattern to match JSON
    json_pattern = r'\{(?:[^{}]|(?R))*\}'

    # Find all matches of JSON in the text
    matches = re.findall(json_pattern, model_pred)

    # Parse the JSON data
    json_data = [json.loads(match) for match in matches]

    return json_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="TheBloke/Llama-2-7B-Chat-fp16")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./muc")
    parser.add_argument("--output-path", type=str, default="./muc_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # Get test file
    if not '.jsonl' in args.data_path:
        fp = os.path.join(args.data_path, 'test.jsonl')
    elif os.path.exists(args.data_path):
        fp = args.data_path
    else:
        raise ValueError(f"Invalid data path: {args.data_path}")
    if not os.path.exists(fp):
        download_url(
            'https://raw.githubusercontent.com/GithuBarry/DocIE-Probing/0067d0e34d15ab37912026e6ff07e22e2f39b084/Corpora/MUC/muc/processed/test.json',
            args.data_path)
        os.rename(os.path.join(args.data_path, 'test.json'), fp)

    list_data_dict = load_jsonl(fp, input='doctext')

    if args.debug:
        list_data_dict = list_data_dict[:10]

    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    # llm.set_stop_words(["Q:", "\end{code}"])
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.0
    elif len(early_exit_layers) == 2:
        print(
            f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
        mode = "dola-static"
        mature_layer = early_exit_layers[1]
        premature_layer = early_exit_layers[0]
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    else:
        print(
            f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mode = "dola"
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {l: 0 for l in candidate_premature_layers}
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    answers = []
    result_dict = defaultdict(list)
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample['input'])
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p,
                               top_k=args.top_k, temperature=args.temperature,
                               repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer,
                               premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers,
                               relative_top=args.relative_top)
        model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        print("model_completion", model_completion)
        print("-----")
        if mode == "dola":
            for k, v in c_dist.items():
                premature_layer_dist[k] += v
        result_dict['model_completion'].append(model_completion)
        result_dict['full_input_text'].append(input_text)
        result_dict['mode'].append(mode)
        result_dict['early_exit_layers'].append(early_exit_layers)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')

    if mode == "dola" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(
                    premature_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path + "_" + str(args.shard_id) + ".json")
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
