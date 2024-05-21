import os
import re
import json
import argparse
from string import Template

from utils import openai_chat_completions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-temp", "--template_path", type=str, default="./prompts/EvaluationV3.txt")
    parser.add_argument("-api", "--api_data_path", type=str, default="")
    parser.add_argument("-gold", "--golden_answer_path", type=str, default="")
    parser.add_argument("-out", "--output_path", type=str, default="")
    parser.add_argument("--continue_run", action="store_true", default=False)
    args = parser.parse_args()

    template = Template(open(args.template_path, "r").read())
    golden_data_info = json.load(open('/home/huan/projects/ToolAlpaca/golden_correct_mix.json'))
    api_data = json.load(open(args.api_data_path, "r"))
    if os.path.exists(args.golden_answer_path):
        golden_answer = json.load(open(args.golden_answer_path, "r"))
        for k, v in zip(api_data, golden_answer):
            k["Golden_Answers"] = v["Golden_Answers"]

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


    golden_answer = json.load(open('./golden-eval_mix.json', "r"))

    index = 0
    invalid_api_count = 0
    for api_idx, api in enumerate(api_data):
        api_name = api.get("Name", api.get("API"))
        if exist_ids is None or api_name not in exist_ids:
            original_data[api_name] = []
        for ques_id, ques in enumerate(api["Instructions"]):
            if {'api': api['Name'], 'idx': ques_id} not in golden_data_info:
                invalid_api_count += 1
                continue
            if exist_ids is not None and ques_id in exist_ids.get(api_name, []):
                continue
            if 'error' in api["Instances"][ques_id].keys() or \
                    "intermediate_steps" not in api["Instances"][ques_id] or \
                    len(api["Instances"][ques_id]["intermediate_steps"]) == 0 or \
                    'Could not parse LLM output:' in api["Instances"][ques_id]['output']:
                original_data["statistics"]["error_num"] += 1
                tmp = {
                    "id": ques_id,
                    "input": "",
                    "output": ""
                }
                # error_samples.append((api_name, tmp['id']))
                print(api_name, tmp['id'])
                original_data[api_name].append(tmp)
                continue
            original_data["statistics"]["num"] += 1

            standard_answer = ""
            for ans_id, ans in enumerate(golden_answer[api_idx]['Instances'][ques_id]['intermediate_steps']):
                standard_answer += f"{ans_id + 1}. Function: {ans[0][0]}\nParameters: {ans[0][1]}\n" \
                                   # f"Responses: {ans[1]}\n"

            solution = ""
            for sol_id, sol in enumerate(api["Instances"][ques_id]["intermediate_steps"]):
                solution += f"{sol_id + 1}. Function: {sol[0][0]}\nParameters: {sol[0][1]}\nResponses: {sol[1]}\n"
            # solution += f"{sol_id + 2}. Final Response: {api_info['Instances'][ques_id]['output']}"

            prompt = template.substitute(
                documentation=api["NLDocumentation"],
                instruction=ques,
                standard=standard_answer,
                solution=solution
            )

            prompt = [{"role": "user", "content": prompt}]
            output = openai_chat_completions(prompt, model="gpt-4-0613", temperature=0.2)
            print(index)
            index = index + 1
            text = output["choices"][0]["message"]["content"]

            results_text = text.split('## Results', 1)[-1]

            process_correctness_match = re.search('Process Correctness: (\w+)', results_text)

            process_correctness_word = process_correctness_match.group(1) if process_correctness_match else ""

            final_response_correctness_match = re.search('Response Correctness: (\w+)', results_text)

            final_response_correctness_word = final_response_correctness_match.group(1) if final_response_correctness_match else ""

            tmp = {
                "id": ques_id,
                "input": prompt,
                "output": text,
                "process_correctness": process_correctness_word,
                "response_correctness": final_response_correctness_word
            }

            original_data["statistics"]["process"][process_correctness_word] += 1
            original_data["statistics"]["response"][final_response_correctness_word] += 1
            if process_correctness_word == final_response_correctness_word == "Yes":
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