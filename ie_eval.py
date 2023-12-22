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
import transformers
from tqdm import tqdm

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
    return (
        "<s>[INST] <<SYS>> Output in JSON only without additional texts. "
        "Use fewer linebreaks in output. <</SYS>> "
        f"Original text: '{input_text}'\n"
        "-----\nGiven above text, "
        "there might be zero or more incidents of types "
        "['kidnapping', 'attack', 'bombing', 'robbery', 'arson', 'forced work stoppage']. "
        "Output a list of incidents. Describe PerpInd (Perpetrator Individual), PerpOrg (Perpetrator Organization), Target (non-people), Victim (people), Weapon"
        "For every incident, fill one empty template dict "
        "{\"incident_type\": '', \"PerpInd\": [], \"PerpOrg\": [], \"Target\": [], \"Victim\": [], \"Weapon\": []}. "
        "Filler string must be short noun phrase exactly from original text and related to the incident. One string per entity."
        "incident_type must one of the element from the list. "
        "Strictly one template dict per incident, if any. Do not output an incident twice even if it fits two or more incident types"
        "Use only one template for an incident that involves multiple PerpOrg, PerpInd, Weapon, Victim, Target."
        "The output MUST be in valid JSON, a list of above template dictionary. "
        "[/INST] Here are the incidents described in the text in JSON:")


def build_prompt_one_event(input_text):
    return (
        "<s>[INST] <<SYS>> Output in JSON only without additional texts. "
        "Use fewer linebreaks in output. <</SYS>> "
        f"Original text: '{input_text}'\n"
        "-----\nGiven above text, "
        "there might be zero or one incident of types"
        "['kidnapping', 'attack', 'bombing', 'robbery', 'arson', 'forced work stoppage']. "
        "Output one or zero incident. Describe PerpInd (Perpetrator Individual), PerpOrg (Perpetrator Organization), Target (non-people), Victim (people), Weapon"
        "If there is an incident, fill the empty template dict "
        "{\"incident_type\": '', \"PerpInd\": [], \"PerpOrg\": [], \"Target\": [], \"Victim\": [], \"Weapon\": []}. "
        "Filler string must be short noun phrase exactly from original text and related to the incident. One string per entity."
        "incident_type must one of the element from the list. "
        "The output MUST be in valid JSON template dictionary or an empty JSON if none exists. "
        "[/INST] Here are the incidents described in the text in JSON:")


def build_prompt_few_shot(input_text):
    return (
        "<s>[INST]<<SYS>> Extract terrorist events only. `incident_type` must be one of "
        "['kidnapping', 'attack', 'bombing', 'robbery', 'arson', 'forced work stoppage'][/INST]"
        "<</SYS>>\n"
        "INPUT: official sources have reported that several guerrilla attacks and heavy fighting took place the "
        "evening of 9 january and this morning throughout the country, and as a result, three soldiers were killed "
        "and three others injured.    alleged guerrilla urban commandos launched two highpower bombs against a car "
        "dealership in downtown san salvador this morning.  a police report said that the attack set the building on "
        "fire, but did not result in any casualties although economic losses are heavy.    during the evening of 9 "
        "january, guerrilla urban commandos bombed two electricity facilities in different places in san salvador, "
        "which caused power outages in some areas of the capital.    meanwhile, the armed forces press committee ("
        "coprefa) reported today that three army soldiers were killed recently in clashes against members of the "
        "farabundo marti national liberation front (fmln) in different parts of the central and eastern regions of "
        "the country.    the war bulletin by coprefa stated that the clashes, in which three members of the general "
        "juan ramon belloso battalion were injured, took place in san jose guayabal, in the central cuscatlan "
        "department, and in santa elena in the eastern usulutan department.\n\n"
        'OUTPUT: [{"incident_type": "bombing", "PerpInd": [["guerrilla urban commandos"]], "PerpOrg": [], '
        '"Target": [["car dealership"]], "Victim": [], "Weapon": [["highpower bombs"]]}, '
        '{"incident_type": "bombing", "PerpInd": [["guerrilla urban commandos"]], "PerpOrg": [], "Target": [['
        '"electricity facilities"], "Victim": [], "Weapon": []}]\n'
        "----\n"
        "INPUT: a war bulletin indicates that on 6 january at 1625, fmln (farabundo marti national liberation front) "
        "troops clashed with the cavalry company in finca santa elena, santa tecla, near san salvador, killing three "
        "and wounding four enemy troops, including the patrol leader who was among those killed.  our troops seized "
        "an m-14 rifle from the enemy, 3,000 cartridges for a 7.62-mm rifle, five knapsacks, six grenades, "
        "and field equipment.\n\n"
        "OUTPUT: []\n"
        "----\n"
        "INPUT: here is an official defense ministry communique:    1. at 0945 this morning, a group of subversives "
        "conducted an armed terrorist attack against former defense minister divison general enrique lopez albujar, "
        "retired.    2. the victim was taken to the air force hospital where he unfortunately passed away.    3. the "
        "authorities are investigating the attack, and they are conducting the appropriate operations to capture the "
        "criminals.    (dated and signed) lima, 9 january 1990.  defense ministry communications office.\n\n"
        'OUTPUT: [{"incident_type": "attack", "PerpInd": [["group of subversives", "subversives"]], '
        '"PerpOrg": [], "Target": [], "Victim": [["enrique lopez albujar"]], "Weapon": []}]\n'
        "----\n"
        f"INPUT: '{input_text}'\n\n"
        "OUTPUT:")


def build_prompt_few_shot_1max(input_text):
    return (
        "<s>[INST]<<SYS>> Extract terrorist events only. `incident_type` must be one of "
        "['kidnapping', 'attack', 'bombing', 'robbery', 'arson', 'forced work stoppage'][/INST]"
        "<</SYS>>\n"
        "INPUT: a war bulletin indicates that on 6 january at 1625, fmln (farabundo marti national liberation front) "
        "troops clashed with the cavalry company in finca santa elena, santa tecla, near san salvador, killing three "
        "and wounding four enemy troops, including the patrol leader who was among those killed.  our troops seized "
        "an m-14 rifle from the enemy, 3,000 cartridges for a 7.62-mm rifle, five knapsacks, six grenades, "
        "and field equipment.\n\n"
        "OUTPUT: []\n"
        "----\n"
        "INPUT: here is an official defense ministry communique:    1. at 0945 this morning, a group of subversives "
        "conducted an armed terrorist attack against former defense minister divison general enrique lopez albujar, "
        "retired.    2. the victim was taken to the air force hospital where he unfortunately passed away.    3. the "
        "authorities are investigating the attack, and they are conducting the appropriate operations to capture the "
        "criminals.    (dated and signed) lima, 9 january 1990.  defense ministry communications office.\n\n"
        'OUTPUT: [{"incident_type": "attack", "PerpInd": [["group of subversives", "subversives"]], '
        '"PerpOrg": [], "Target": [], "Victim": [["enrique lopez albujar"]], "Weapon": []}]\n'
        "----\n"
        f"INPUT: '{input_text}'\n\n"
        "OUTPUT:")


def build_prompt_combined(input_text):
    return (
        "<s>[INST]<<SYS>> Given above text, "
        "there might be zero or more incidents of types "
        "['kidnapping', 'attack', 'bombing', 'robbery', 'arson', 'forced work stoppage']. "
        "Output a list of incidents. Describe PerpInd (Perpetrator Individual), PerpOrg (Perpetrator Organization), Target (non-people), Victim (people), Weapon"
        "For every incident, fill one empty template dict "
        "{\"incident_type\": '', \"PerpInd\": [], \"PerpOrg\": [], \"Target\": [], \"Victim\": [], \"Weapon\": []}. "
        "Filler string must be short noun phrase exactly from original text and related to the incident. One string per entity."
        "incident_type must one of the element from the list. "
        "Strictly one template dict per incident, if any. Do not output an incident twice even if it fits two or more incident types"
        "Use only one template for an incident that involves multiple PerpOrg, PerpInd, Weapon, Victim, Target.[/INST]"
        "<</SYS>>\n"
        "INPUT: official sources have reported that several guerrilla attacks and heavy fighting took place the "
        "evening of 9 january and this morning throughout the country, and as a result, three soldiers were killed "
        "and three others injured.    alleged guerrilla urban commandos launched two highpower bombs against a car "
        "dealership in downtown san salvador this morning.  a police report said that the attack set the building on "
        "fire, but did not result in any casualties although economic losses are heavy.    during the evening of 9 "
        "january, guerrilla urban commandos bombed two electricity facilities in different places in san salvador, "
        "which caused power outages in some areas of the capital.    meanwhile, the armed forces press committee ("
        "coprefa) reported today that three army soldiers were killed recently in clashes against members of the "
        "farabundo marti national liberation front (fmln) in different parts of the central and eastern regions of "
        "the country.    the war bulletin by coprefa stated that the clashes, in which three members of the general "
        "juan ramon belloso battalion were injured, took place in san jose guayabal, in the central cuscatlan "
        "department, and in santa elena in the eastern usulutan department.\n\n"
        'OUTPUT: [{"incident_type": "bombing", "PerpInd": [["guerrilla urban commandos"]], "PerpOrg": [], '
        '"Target": [["car dealership"]], "Victim": [], "Weapon": [["highpower bombs"]]}, '
        '{"incident_type": "bombing", "PerpInd": [["guerrilla urban commandos"]], "PerpOrg": [], "Target": [['
        '"electricity facilities"], "Victim": [], "Weapon": []}]\n'
        "----\n"
        "INPUT: a war bulletin indicates that on 6 january at 1625, fmln (farabundo marti national liberation front) "
        "troops clashed with the cavalry company in finca santa elena, santa tecla, near san salvador, killing three "
        "and wounding four enemy troops, including the patrol leader who was among those killed.  our troops seized "
        "an m-14 rifle from the enemy, 3,000 cartridges for a 7.62-mm rifle, five knapsacks, six grenades, "
        "and field equipment.\n\n"
        "OUTPUT: []\n"
        "----\n"
        "INPUT: here is an official defense ministry communique:    1. at 0945 this morning, a group of subversives "
        "conducted an armed terrorist attack against former defense minister divison general enrique lopez albujar, "
        "retired.    2. the victim was taken to the air force hospital where he unfortunately passed away.    3. the "
        "authorities are investigating the attack, and they are conducting the appropriate operations to capture the "
        "criminals.    (dated and signed) lima, 9 january 1990.  defense ministry communications office.\n\n"
        'OUTPUT: [{"incident_type": "attack", "PerpInd": [["group of subversives", "subversives"]], '
        '"PerpOrg": [], "Target": [], "Victim": [["enrique lopez albujar"]], "Weapon": []}]\n'
        "----\n"
        f"INPUT: '{input_text}'\n\n"
        "OUTPUT:")


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
    parser.add_argument("--max_gpu_memory", type=int, default=45)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./muc")
    parser.add_argument("--output-path", type=str, default="./muc_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--do_shuffle", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--prompt_scheme", type=str, choices=["oneevent", "default", "fewshot", "fewshot_more", "combined"],
                        default="default")
    args = parser.parse_args()
    print("args", args)
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    output_file = (args.output_path + "_" + args.prompt_scheme + "_" + args.model_name.split("/")[
        -1] + "_" + args.early_exit_layers + ".json")

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
        if args.prompt_scheme == "default":
            input_text = build_prompt(sample['input'])
        elif args.prompt_scheme == "oneevent":
            input_text = build_prompt_one_event(sample['input'])
        elif args.prompt_scheme == "fewshot":
            llm.set_stop_words(["INPUT:"])
            input_text = build_prompt_few_shot_1max(sample['input'])
        elif args.prompt_scheme == "fewshot_more":
            llm.set_stop_words(["INPUT:"])
            input_text = build_prompt_few_shot(sample['input'])
        elif args.prompt_scheme == "combined":
            llm.set_stop_words(["INPUT:"])
            input_text = build_prompt_combined(sample['input'])
        else:
            raise Exception
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p,
                               top_k=args.top_k, temperature=args.temperature,
                               repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer,
                               premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers,
                               relative_top=args.relative_top)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        print("-----")
        if mode == "dola":
            for k, v in c_dist.items():
                premature_layer_dist[k] += v
        result_dict['model_completion'].append(model_completion)
        result_dict['full_input_text'].append(input_text)
        result_dict['mode'].append(mode)
        result_dict['early_exit_layers'].append([int(x) for x in args.early_exit_layers.split(',')])

        with open(output_file, 'w') as f:
            json.dump(result_dict, f)

    if mode == "dola" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(
                    premature_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
