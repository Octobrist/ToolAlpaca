import os
import json
import logging
import argparse

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

final_output_path = os.path.join(args.output_dir, f"mutil-api/{args.llm.split('/')[-1]}_mix.json")

if args.use_cache:
    res = requests.get(f"{args.server_url}/__simulator_cache__/open")

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
                if {'api': api['Name'], 'idx': idx, 'steps':len(api['Golden_Answers'][idx])} not in golden_data_info:
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
                try:
                    generate_count += 1
                    output = agent(
                        {
                            'input': inst,
                            'intermediate_steps': pred_steps
                        }
                    )
                    json.dumps(output, ensure_ascii=4)
                except json.JSONDecodeError:
                    output = {'error': str(output)}
                except Exception as e:
                    logger.error(e)
                    output = {"error": str(e)}
                # output = {}
                output['times'] = last_times
                output['times'] = output['times'] + 1
                if args.use_cache:
                    res = requests.get(f"{args.server_url}/__simulator_cache__/clear/{api['Name']}")
                    print(res.text)
                Answers.append(output)
            api_data[api_idx]['Instances'] = Answers
            assert len(api_data[api_idx]['Instances']) == len(api_data[api_idx]['Golden_Answers'])
    print('genertate_count: ', generate_count)

json.dump(
    api_data,
    open(final_output_path, "w", encoding="utf-8"),
    indent=4,
    ensure_ascii=False
)

