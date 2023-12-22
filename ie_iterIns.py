import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

s = requests.Session()

api_base = os.getenv("OPENAI_BASE_URL")
token = os.getenv("OPENAI_API_KEY")
url = f"{api_base}/chat/completions"


def query(sys, user):
    result = None
    while not result:
        try:
            body = {
                "model": "meta-llama/Llama-2-70b-chat-hf",
                "messages": [{"role": "system", "content": sys},
                             {"role": "user", "content": user}],
                "temperature": 0.7
            }

            with s.post(url, headers={"Authorization": f"Bearer {token}"}, json=body) as resp:
                result = json.loads(resp.content.decode())['choices'][0]['message']['content']
        except Exception:
            pass
    return result


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
    parser.add_argument("--data-path", type=str, default="./muc")
    parser.add_argument("--output-path", type=str, default="./muc_result")
    parser.add_argument("--prompt_scheme", type=str,
                        choices=["oneevent", "default", "fewshot", "fewshot_more", "combined"],
                        default="default")
    args = parser.parse_args()
    print("args", args)

    output_file = (args.output_path + "_" + args.prompt_scheme + "_" + "LLAMA2-70B.json")

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

    answers = []
    result_dict = defaultdict(list)


    def process_sample(sample):
        if args.prompt_scheme == "default":
            input_text = build_prompt(sample['input'])
        elif args.prompt_scheme == "oneevent":
            input_text = build_prompt_one_event(sample['input'])
        elif args.prompt_scheme == "fewshot":
            input_text = build_prompt_few_shot_1max(sample['input'])
        elif args.prompt_scheme == "fewshot_more":
            input_text = build_prompt_few_shot(sample['input'])
        elif args.prompt_scheme == "combined":
            input_text = build_prompt_combined(sample['input'])
        else:
            raise Exception

        # Extracting text between <<SYS>> and <</SYS>>
        system = re.findall('<<SYS>>(.*?)<</SYS>>', input_text, re.DOTALL)
        system = ' '.join(system)  # Concatenate if multiple system texts

        # Removing the system text from the input_text
        user = re.sub('<<SYS>>.*?<</SYS>>', '', input_text, flags=re.DOTALL).replace("[INST]", "").replace("[/INST]",
                                                                                                           "")

        model_completion = query(system, user)

        return model_completion, input_text


    # Use ThreadPoolExecutor to process samples concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks
        future_to_sample = {executor.submit(process_sample, sample): sample for sample in list_data_dict}

        # Process results as they complete
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                model_completion, input_text = future.result()
                print("-----")
                result_dict['model_completion'].append(model_completion)
                result_dict['full_input_text'].append(input_text)
            except Exception as exc:
                print(f"Sample {sample} generated an exception: {exc}")

    # save results to a json file
    model_tag = "llama2-70B"
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
