import json
import re
from collections import defaultdict

if __name__ == '__main__':
    counter = defaultdict(lambda: 0)
    with open("./OriginalOutput/muc_result-dola-json.json") as f:
        output_j = json.load(f)
    examples = [*zip(output_j['model_completion'], output_j['full_input_text'], output_j['mode'],
                     output_j['early_exit_layers'])]
    assert len(examples) == 200
    for i, example in enumerate(examples):
        model_output = example[0]
        json_objects = re.findall(r'\{[^{}]*\}', model_output)

        parsed_incidents = []
        for json_str in json_objects:
            try:
                incident = json.loads(json_str)
                parsed_incidents.append(incident)
                counter['Valid incident'] += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON:{i} {e}")
                counter['Err'] += 1
        examples[i] = {"parsed_incidents": parsed_incidents, "model_completion": example[0],
                       "full_input_text": example[1], "mode": example[2],
                       "early_exit_layers": example[3]}

    print(examples)
