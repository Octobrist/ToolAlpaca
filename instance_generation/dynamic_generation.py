import os
import json
import logging
import argparse
import re
import openai
import requests
from langchain.schema import AgentAction, AgentFinish
from tqdm import tqdm
from langchain.llms import OpenAI
from langchain import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from agent.get_agent import get_agent
from agent.agent_prompts import prompt_proj
from utils import load_openapi_spec, analyze_openapi_spec
from feedback.dynamic_feedback import get_incorrect_samples, dynamic_feedback, get_cur_details, get_cur_details_from_last_feedbacks
from feedback.prompt import *
import torch
import deepspeed

logger = logging.getLogger(__name__)

ober_list = []
output_list = []

def get_nonnone_feedbacks(last_feedbacks):
    new_feedbacks = []
    for feedback in last_feedbacks:
        if feedback[0][0] is None and feedback[0][1] == 'None':
            continue
        new_feedbacks.append(feedback)
    return new_feedbacks

def get_cur_intermediate_steps(pre_steps, action, action_input, action_observation, action_output, feedback_type):
    ober_list.append(action_observation)
    output_list.append(action_output)
    cur_oberservation = None
    if 'NoneType' in action_output:
        assert action_observation is None
        cur_oberservation = f"ERROR. {action_output}"
    elif 'Could not parse LLM output' in action_output:
        assert action_observation is None and action is None and action_input is None
        cur_oberservation = f"ERROR. {action_output}"
    elif action_output == 'Agent stopped due to iteration limit or time limit.': # 正常
        assert action is not None and action_input is not None
        cur_oberservation = action_observation
        # if 'Invalid JSON format.' not in action_observation:
        #     print(2)
        # else:
        #     print(3)
    elif 'ASSISTANT Response:' in action_output:
        return None
    else:
        raise KeyError
    # elif action_output == 'Agent stopped due to iteration limit or time limit.':
    #     action_observation = "This action does not collaborate with known API calls to facilitate the resolution of user's task instructions. " \
    #                          "I would like to independently regenerate a new Action and a new Action Input. "
    # else:
    #     print(action_observation)
    cur_step = [[action, str(action_input)], cur_oberservation]
    return cur_step

def get_observation_and_output():
    return set(ober_list), set(output_list)

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
parser.add_argument("--dynamic_type", type=str, default='normal')
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

final_output_path = os.path.join(args.output_dir, f"single-api-dynamic/{args.input_data_path.split('/')[-1].replace('.json', '')}_{args.dynamic_type}_epoch{args.epoch}_wo_cot.json")

if args.use_cache:
    res = requests.get(f"{args.server_url}/__simulator_cache__/open")

count = 0
regenerte_count = 0
cur_step_list = []
for api_idx, api in tqdm(enumerate(api_data)):
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
            if 'epoch' not in args.input_data_path:
                eval_info = json.load(open(args.input_data_path.replace('/generate/single-api/', '/eval/v3/single-api/'),'r'))
            else:
                if 'dynamic' in args.input_data_path:
                    eval_info = json.load(open(args.input_data_path.replace('/generate/single-api-dynamic/', '/eval/v3/single-api-dynamic/'), 'r'))
                else:
                    eval_info = json.load(open(args.input_data_path.replace('/generate/single-api-static/v3/',  '/eval/v3/single-api-static/'), 'r'))
            incorrect_samples = get_incorrect_samples(eval_info)
            if len(api.get("Authentication", [])) > 0:
                inst += "\nAuthentication information: " + \
                      " ".join([f"{k}={v}" for k, v in api["Authentication"].items()])
            api_docs = get_description(api['Function_Description'])
            outputs = {}
            for cur_step_idx in range(len(api['Golden_Answers'][idx])):
                reserve_feedback_idx = 0
                if (api['Name'], f'{idx}|{cur_step_idx}') not in incorrect_samples:
                    continue
                regenerte_count += 1
                if 'last_feedbacks' in api['Instances'][idx][str(cur_step_idx)].keys():
                    api['Instances'][idx][str(cur_step_idx)]['last_feedbacks'] = get_nonnone_feedbacks(api['Instances'][idx][str(cur_step_idx)]['last_feedbacks'])
                pre_steps = get_instance_intermediate_steps(golden_data[api_idx]['Instances'][idx]['intermediate_steps'], cur_step_idx)
                cur_action, cur_action_input, cur_oberservation, cur_action_output = get_cur_details(api['Instances'][idx][str(cur_step_idx)])
                cur_step = get_cur_intermediate_steps(pre_steps, cur_action, cur_action_input, cur_oberservation, cur_action_output, args.dynamic_type)
                if cur_step is None:
                    cur_action, cur_action_input, cur_oberservation = None, None, None
                    if 'last_feedbacks' in api['Instances'][idx][str(cur_step_idx)].keys():
                        cur_action, cur_action_input, reserve_feedback_idx = get_cur_details_from_last_feedbacks(api['Instances'][idx][str(cur_step_idx)]['last_feedbacks'])
                        action = AgentAction(cur_action, cur_action_input, f'\nASSISTANT Action: {cur_action}\nASSISTANT Action Input: {cur_action_input}')
                        cur_oberservation = agent.take_action(action)
                    cur_step = [[cur_action, cur_action_input], cur_oberservation]
                assert cur_step is not None
                try:
                    output = agent(
                        {
                            'dynamic': True,
                            'input': inst,
                            'intermediate_steps':pre_steps,
                            'cur_step': cur_step,
                            'last_feedbacks': None if 'last_feedbacks' not in api['Instances'][idx][str(cur_step_idx)].keys() else api['Instances'][idx][str(cur_step_idx)]['last_feedbacks'][:len(api['Instances'][idx][str(cur_step_idx)]['last_feedbacks']) - reserve_feedback_idx],
                            'dynamic_feedbacks': None if 'dynamic_feedbacks' not in api['Instances'][idx][str(cur_step_idx)].keys() else api['Instances'][idx][str(cur_step_idx)]['dynamic_feedbacks']
                        }
                    )
                    output['last_feedbacks'] = [] if 'last_feedbacks' not in api['Instances'][idx][str(cur_step_idx)].keys() else api['Instances'][idx][str(cur_step_idx)]['last_feedbacks'][:len(api['Instances'][idx][str(cur_step_idx)]['last_feedbacks']) - reserve_feedback_idx]
                    output['dynamic_feedbacks'] = [] if 'dynamic_feedbacks' not in api['Instances'][idx][str(cur_step_idx)].keys() else api['Instances'][idx][str(cur_step_idx)]['dynamic_feedbacks']
                    output['dynamic_feedbacks'].append(cur_step)
                    output.pop('cur_step')
                    output.pop('dynamic')
                    json.dumps(output, ensure_ascii=4)
                except json.JSONDecodeError:
                    output = str(output)
                except Exception as e:
                    logger.error(e)
                    output = {"error": str(e)}

                if args.use_cache:
                    res = requests.get(f"{args.server_url}/__simulator_cache__/clear/{api['Name']}")
                    print(res.text)

                if 'intermediate_steps' not in output.keys() or len(output['intermediate_steps']) == len(pre_steps):
                    print('???')
                    output['intermediate_steps'].append(output['dynamic_feedbacks'][-1])
                    api_data[api_idx]['Instances'][idx][str(cur_step_idx)] = output
                elif len(output['intermediate_steps']) == len(pre_steps) + 1:
                    print('?')
                    api_data[api_idx]['Instances'][idx][str(cur_step_idx)] = output
                else:
                    raise KeyError
with open(final_output_path , 'w') as file:
    json.dump(api_data, file, indent=4)

# s1, s2 = get_observation_and_output()
print(count)
print(regenerte_count)