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
    elif 'ASSISTANT Response' in action_output: # 不需要反馈了？
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
parser.add_argument("-sou", "--source", type=str, required=True)
parser.add_argument("-out", "--output_dir", type=str, required=True)
args = parser.parse_args()


import shutil
import os

source_file = args.source
api_data = json.load(open(source_file , "r"))
incorrect_samples = get_incorrect_samples(api_data)
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
    list_value = list(values)
    for value in list_value:
        if (key, value['id']) in incorrect_samples:
            original_data[key].remove(value)
            if 'process_correctness'in value.keys():
                original_data['statistics']['num'] -= 1
                original_data['statistics']['process'][value['process_correctness']] -= 1
                original_data['statistics']['response'][value['response_correctness']] -= 1
            else:
                original_data['statistics']['error_num'] -= 1

print(args.output_dir)
# with open(args.output_dir, 'w') as file:
#     json.dump(original_data, file, indent=4)


