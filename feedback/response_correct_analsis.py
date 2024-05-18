import os
import re
import json
import argparse
from string import Template

from utils import openai_chat_completions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-temp", "--template_path", type=str, default="./prompts/EvaluationV2.txt")
    parser.add_argument("-api", "--api_data_path", type=str, default="")
    parser.add_argument("-gold", "--golden_answer_path", type=str, default="./golden-eval_real.json")
    parser.add_argument("-out", "--output_path", type=str, default="")
    parser.add_argument("--continue_run", action="store_true", default=False)
    args = parser.parse_args()

    golden_data_info = json.load(open('/home/huan/projects/ToolAlpaca/golden_correct.json'))
    template = Template(open(args.template_path, "r").read())

    api_data = json.load(open(args.api_data_path, "r"))
    if os.path.exists(args.golden_answer_path):
        golden_answer = json.load(open(args.golden_answer_path, "r"))

    invalid_api_count = 0
    original_data = {}
    original_data["statistics"] = {
        "num": 0,
        "error_num": 0,
        "process": {
            "Yes": 0,
            "No": 0,
            "Uncertain": 0
        },
        "response": {
            "Yes": 0,
            "No": 0,
            "Uncertain": 0
        },
        "both": 0
    }

    exist_ids = None
    if args.continue_run:
        original_data = json.load(open(os.path.join(args.output_path, args.api_data_path.split('/')[-1]), "r"))
        exist_ids = {i: [j["id"] for j in original_data[i]] for i in original_data if i != "statistics"}
    error_samples = []

    index = 0
    for api_idx, api in enumerate(api_data):
        api_name = api.get("Name", api.get("API"))
        if exist_ids is None or api_name not in exist_ids:
            original_data[api_name] = []
        for ques_id, ques in enumerate(api["Instructions"]):
            if {'api': api['Name'], 'idx': ques_id, 'steps': len(api['Golden_Answers'][ques_id])} not in golden_data_info:
                invalid_api_count += 1
                continue
            assert len(api['Golden_Answers'][ques_id]) == len(golden_answer[api_idx]['Instances'][ques_id]['intermediate_steps'])
            for cur_step in range(len(api['Golden_Answers'][ques_id])):
                if exist_ids is not None and f'{ques_id}|{cur_step}' in exist_ids.get(api_name, []):
                    continue
                if 'error' in api["Instances"][ques_id][str(cur_step)].keys() or \
                        "intermediate_steps" not in api["Instances"][ques_id][str(cur_step)] or \
                        len(api["Instances"][ques_id][str(cur_step)]["intermediate_steps"]) <= cur_step or \
                        'Could not parse LLM output:' in api["Instances"][ques_id][str(cur_step)]['output']:
                    original_data["statistics"]["error_num"] += 1
                    tmp = {
                        "id": str(ques_id) + '|' + str(cur_step),
                        "input": "",
                        "output": ""
                    }
                    error_samples.append((api_name, tmp['id']))
                    print(api_name, tmp['id'])
                    original_data[api_name].append(tmp)
                    continue

                original_data["statistics"]["num"] += 1
                pre_step_idx = 0
                prefix_answer = ""
                for pre_step in golden_answer[api_idx]['Instances'][ques_id]['intermediate_steps'][:cur_step]:
                    prefix_answer += f"{pre_step_idx + 1}. Function: {pre_step[0][0]}\nParameters: {pre_step[0][1]}\nResponses: {pre_step[1]}\n"
                    pre_step_idx = pre_step_idx + 1

                standard_answer = ""
                ans = golden_answer[api_idx]['Instances'][ques_id]['intermediate_steps'][cur_step]
                standard_answer = f"{pre_step_idx + 1}. Function: {ans[0][0]}\nParameters: {ans[0][1]}\nResponses: {ans[1]}\n"

                solution = ""
                sol = api["Instances"][ques_id][str(cur_step)]["intermediate_steps"][cur_step]
                solution += f"{pre_step_idx + 1}. Function: {sol[0][0]}\nParameters: {sol[0][1]}\nResponses: {sol[1]}\n"

                prompt = template.substitute(
                    documentation=api["NLDocumentation"],
                    instruction=ques,
                    standard=standard_answer,
                    solution=solution,
                    prefix=prefix_answer
                )

                prompt = [{"role": "user", "content": prompt}]
                output = openai_chat_completions(prompt, model="gpt-4-0613", temperature=0.2)
                print(index)
                index = index + 1
                text = output["choices"][0]["message"]["content"]

                results_text = text.split('## Results', 1)[-1]

                process_correctness_match = re.search('Process Correctness: (\w+)', results_text)

                process_correctness_word = process_correctness_match.group(1) if process_correctness_match else ""

                response_correctness_match = re.search('Response Correctness: (\w+)', results_text)

                response_correctness_word = response_correctness_match.group(1) if response_correctness_match else ""

                tmp = {
                    "id": str(ques_id) + '|' + str(cur_step),
                    "input": prompt,
                    "output": text,
                    "process_correctness": process_correctness_word,
                    "response_correctness": response_correctness_word
                }

                original_data["statistics"]["process"][process_correctness_word] += 1
                original_data["statistics"]["response"][response_correctness_word] += 1
                if process_correctness_word == response_correctness_word == "Yes":
                    original_data["statistics"]["both"] += 1

                original_data[api_name].append(tmp)

                json.dump(
                    original_data,
                    open(os.path.join(args.output_path, args.api_data_path.split('/')[-1]), "w"),
                    indent=4,
                    ensure_ascii=False
                )
    json.dump(
        original_data,
        open(os.path.join(args.output_path, args.api_data_path.split('/')[-1]), "w"),
        indent=4,
        ensure_ascii=False
    )
    print(invalid_api_count)
    print(original_data["statistics"]["num"])
    print(original_data["statistics"]["error_num"])

    # import pickle
    # with open("./error_samples.pkl", 'wb') as file:
    #     pickle.dump(error_samples, file)