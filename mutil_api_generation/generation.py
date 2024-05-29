import os
import re
import json
import logging
import argparse
from copy import deepcopy

import openai
import requests
from tqdm import tqdm
from langchain.llms import OpenAI
from langchain import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from agent.get_agent import get_agent
from agent.agent_prompts import prompt_proj
from utils import load_openapi_spec, analyze_openapi_spec
import torch
import deepspeed

logger = logging.getLogger(__name__)

def match_response(text):
    status_code_match = re.search(r'Status Code: (\d+)', text)
    response_match = re.search(r'Response: (\{.*\})', text)
    if status_code_match and response_match:
        if status_code_match.group(1) == '200' or status_code_match.group(1) == '201' \
                or status_code_match.group(1) == '202' or status_code_match.group(1) == '203' \
            or status_code_match.group(1) == '204' or status_code_match.group(1) == '205':
            return f"Status Code: {status_code_match.group(1)}. Response:{response_match.group(1)}"
        else:
            return f"Status Code: {status_code_match.group(1)}."
    else:
        return ""

def get_instance_intermediate_steps(input_data, step):
    pre_steps = input_data[:step]
    return pre_steps

def judge_all_finish(api_data):
    for api in api_data:
        if 'Instances' not in api.keys():
            return False
        for idx, instance in enumerate(api['Instances']):
            if instance == 'This instance is not used.':
                continue
            if 'times' not in instance.keys():
                return False
            if instance['times'] < len(api['Golden_Answers'][idx]):
                return False
    return True

parser = argparse.ArgumentParser()
parser.add_argument("-api", "--api_data_path", type=str, required=True)
parser.add_argument("-inp", "--input_data_path", type=str, default=None)
parser.add_argument("-out", "--output_dir", type=str, required=True)
parser.add_argument("-llm", type=str, default=None)
parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5678")
parser.add_argument("--agent_prompt", type=str, default="train_v2")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--max_iterations", type=int, default=1)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--use_cache", action="store_true", default=False)
parser.add_argument("--real", action="store_true", default=False)
parser.add_argument("--without_getDetails", action="store_true", default=False)
args = parser.parse_args()

if args.llm is None or args.llm.lower() in ["gpt3", "gpt-3"]:
    llm = OpenAI(temperature=0.0)
elif "chatgpt" in args.llm.lower():
    llm = ChatOpenAI(temperature=0.0)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.llm, trust_remote_code=True).half()
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        device=args.device,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=generator)

api_data = json.load(open(args.api_data_path, "r"))
golden_data_info = json.load(open('/home/huan/projects/ToolAlpaca/golden_correct_mix.json'))

final_output_path = os.path.join(args.output_dir, f"mutil-api-dynamic/{args.llm.split('/')[-1]}_mix_wo_cot.json")
print(final_output_path)
if args.use_cache:
    res = requests.get(f"{args.server_url}/__simulator_cache__/open")

first = True
count = 0
while judge_all_finish(api_data) is False:
    generate_count = 0
    for api_idx, api in tqdm(enumerate(api_data)):
        if "Instructions" not in api or len(api["Instructions"]) == 0:
            continue
        openapi_spec = load_openapi_spec(api["Documentation"])
        input_valid, output_valid = analyze_openapi_spec(openapi_spec)
        if input_valid and output_valid:
            agent = get_agent(
                llm=llm,
                api_data=api,
                server_url=args.server_url,
                agent_prompt=prompt_proj[args.agent_prompt],
                enable_getDetails=not args.without_getDetails,
                max_iterations=args.max_iterations
            )
            Answers = []
            for idx, inst in enumerate(api["Instructions"]):
                if {'api': api['Name'], 'idx': idx} not in golden_data_info:
                    Answers.append('This instance is not used.')
                    continue
                if len(api.get("Authentication", [])) > 0:
                    inst += "\nAuthentication information: " + \
                          " ".join([f"{k}={v}" for k, v in api["Authentication"].items()])
                pred_steps = []
                if 'Instances' in api_data[api_idx].keys():
                    if 'times' in api_data[api_idx]['Instances'][idx].keys():
                        last_times = api_data[api_idx]['Instances'][idx]['times']
                        if api_data[api_idx]['Instances'][idx]['times'] == len(api['Golden_Answers'][idx]):
                            Answers.append(api_data[api_idx]['Instances'][idx])
                            continue
                    else:
                        last_times = 0
                    if 'intermediate_steps' in api_data[api_idx]['Instances'][idx].keys():
                        pred_steps = api_data[api_idx]['Instances'][idx]['intermediate_steps']
                else:
                    last_times = 0
                if first:
                    copy_pred_step = deepcopy(pred_steps) # for sample
                    for step_idx, step in enumerate(copy_pred_step):
                        json_start = min([idx for idx in (step[0][1].find('{'), step[0][1].find('['))
                                          if idx != -1])
                        json_end = max([idx for idx in (step[0][1].rfind('}'), step[0][1].rfind(']'))
                                        if idx != -1]) + 1
                        valid_json_string = step[0][1][json_start:json_end]
                        pred_steps[step_idx][0][1] = valid_json_string
                try:
                    generate_count += 1
                    output = agent(
                        {
                            'input': inst,
                            'intermediate_steps': pred_steps,
                            'cur_step': api_data[api_idx]['Instances'][idx]['cur_step'] if 'Instances' in api_data[api_idx] and 'cur_step' in api_data[api_idx]['Instances'][idx].keys() else None
                        }
                    )
                    json.dumps(output, ensure_ascii=4)
                except json.JSONDecodeError:
                    output = {'error': str(output)}
                except Exception as e:
                    logger.error(e)
                    output = {"error": str(e)}
                output['times'] = last_times
                output['times'] = output['times'] + 1
                if args.use_cache:
                    res = requests.get(f"{args.server_url}/__simulator_cache__/clear/{api['Name']}")
                    print(res.text)
                Answers.append(output)
            api_data[api_idx]['Instances'] = Answers
            assert len(api_data[api_idx]['Instances']) == len(api_data[api_idx]['Golden_Answers'])
    first = False
    print('genertate_count: ', generate_count)
print(final_output_path)
json.dump(
    api_data,
    open(final_output_path, "w", encoding="utf-8"),
    indent=4,
    ensure_ascii=False
)

