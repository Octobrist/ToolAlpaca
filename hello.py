import os
import json
import logging
import argparse
import re
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
from feedback.static_feedback import static_feedback, get_error_details
from feedback.prompt import *
import torch
import deepspeed

logger = logging.getLogger(__name__)


def get_cur_intermediate_steps(pre_steps, action, action_input, prompt):
    cur_step = [[action, str(action_input)], prompt]
    return cur_step
def get_instance_intermediate_steps(input_data, step):
    pre_steps = input_data[:step]
    return pre_steps

def get_description(FC_descs):
    desc = {}
    for FC_name, FC_desc in FC_descs.items():
        if FC_name == "components":
            continue
        # 定义正则表达式模式
        pattern = r'Parameters:\s*(\{.*?\})'
        # 使用正则表达式进行匹配
        match = re.search(pattern, FC_desc)
        # 如果匹配成功，提取Parameters行中的内容
        if match:
            parameters_json = match.group(1)
        else:
            raise ValueError
        desc[FC_name] = eval(parameters_json)
    return desc

parser = argparse.ArgumentParser()
parser.add_argument("-inp", "--input_data_path", type=str, required=True)
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
    llm = ChatOpenAI(temperature=0.0,
    # model_kwargs={"stop": ['\nASSISTANT Observation:', '\n\tASSISTANT Observation:']}
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.llm, trust_remote_code=True).half()
    # model = deepspeed.init_inference(
    #     model=model,  # Transformers models
    #     mp_size=1,  # Number of GPU
    #     dtype=torch.float16,  # dtype of the weights (fp16)
    #     replace_method="auto",  # Lets DS autmatically identify the layer to replace
    #     replace_with_kernel_inject=True,  # replace the model with the kernel injector
    # )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        device=args.device,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=generator)


api_data = json.load(open(args.input_data_path, "r"))
golden_data = json.load(open('/home/huan/projects/ToolAlpaca/golden-eval_real.json'))
golden_data_info = json.load(open('/home/huan/projects/ToolAlpaca/golden_correct.json'))

if args.use_cache:
    res = requests.get(f"{args.server_url}/__simulator_cache__/open")

error_samples = {}

count = 0
total_api = 0
regenerte_count = 0
for api_idx, api in tqdm(enumerate(api_data)):
    error_samples[api['Name']] = []
    assert len(api["Instances"]) == len(api['Golden_Answers'])
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
            if {'api': api['Name'], 'idx': idx, 'steps':len(api['Golden_Answers'][idx])} not in golden_data_info:
                count += 1
                Answers.append('This instance is not used.')
                continue
            if len(api.get("Authentication", [])) > 0:
                inst += "\nAuthentication information: " + \
                      " ".join([f"{k}={v}" for k, v in api["Authentication"].items()])
            api_docs = get_description(api['Function_Description'])
            outputs = {}
            for cur_step_idx in range(len(api['Golden_Answers'][idx])):
                total_api += 1
                prompt = None
                pre_steps = get_instance_intermediate_steps(golden_data[api_idx]['Instances'][idx]['intermediate_steps'], cur_step_idx)
                category, error_type, error_detail, cur_action, cur_action_input = static_feedback(api['Instances'][idx][str(cur_step_idx)], golden_data[api_idx]['Golden_Answers'][idx][cur_step_idx], api_docs)
                if category is None:
                    continue
                elif category == 'no_api_call':
                    prompt = origin_regenerate_no_api_prompts
                elif category == 'api_name_mismatch':
                    prompt = get_regenerate_api_prompts(error_type, error_detail)
                    prompt = prompt.replace('{api_name}', cur_action)
                elif category == 'invalid_parameters':
                    prompt = get_regenerate_key_prompts(error_type, error_detail[1])
                    prompt = prompt.replace('{key_name}', error_detail[0])
                elif category == 'input_mismatch':
                    prompt = get_regenerate_value_prompts(error_type, error_detail[2])
                    prompt = prompt.replace('{pred_key}', error_detail[0])
                    prompt = prompt.replace('{pred_value}', str(error_detail[1]))
                else:
                    raise KeyError
                assert prompt is not None
                regenerte_count += 1
                cur_step = get_cur_intermediate_steps(pre_steps, cur_action, cur_action_input, prompt)
                error_samples[api['Name']].append(f"{idx}|{cur_step_idx}")

ERROR_DETAILS = get_error_details()
for key, value in ERROR_DETAILS.items():
    print(key, value)

print(count)
print(total_api)
print(regenerte_count)


import shutil
import os

source_file = f"./eval/v3/single-api-static/Llama-2-7b-chat-ms_real_epoch3.json"
destination_file = f"./eval/v3/single-api-static/{args.input_data_path.split('/')[-1]}"

api_data = json.load(open(source_file , "r"))

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

for key, values in api_data.items():
    if key == 'statistics':
        original_data[key] = values
        continue
    original_data[key] = values
    for value_idx, value in enumerate(values):
        if value['id'] in error_samples[key]:
            original_data[key].remove(value)
            if 'process_correctness'in value.keys():
                original_data['statistics']['num'] -= 1
                original_data['statistics']['process'][value['process_correctness']] -= 1
                original_data['statistics']['response'][value['response_correctness']] -= 1
            else:
                original_data['statistics']['error_num'] -= 1


with open(destination_file, 'w') as file:
    json.dump(original_data, file, indent=4)


