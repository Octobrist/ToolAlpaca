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

def load_json_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-ori", "--origin_file", type=str, required=True)
args = parser.parse_args()

golden_data = json.load(open('/home/huan/projects/ToolAlpaca/golden_correct_mix.json'))
api_data = load_json_from_file(args.origin_file)

error_detail = {
    '2': [],
    '4': [],
    '5': [],
    'error parser': [], # NO_API_CALL
    'other error': []
}
max_steps = 0
for api_idx, api in tqdm(enumerate(api_data)):
    if "Instructions" not in api or len(api["Instructions"]) == 0:
        continue
    # if len(api["Instructions"]) != len(api['Instances']):
    #     raise ValueError
    openapi_spec = load_openapi_spec(api["Documentation"])
    input_valid, output_valid = analyze_openapi_spec(openapi_spec)
    if input_valid and output_valid:
        Answers = []
        for idx, inst in enumerate(api["Instructions"]):
            ga = api['Golden_Answers'][idx]
            ins = api['Instances'][idx]
            if "intermediate_steps" not in ins.keys():
                error_detail['other error'].append({'api': api['Name'], 'idx': idx})
                continue
            if 'error' in ins.keys():
                error_detail['other error'].append({'api': api['Name'], 'idx': idx, })
                continue
            flag = True
            input = ins['input'] + '\n'
            for intermediate in ins['intermediate_steps']:
                input = input + 'ASSISTANT Action: ' + intermediate[0][0] + '\n'
                input = input + 'ASSISTANT Action Input: ' + intermediate[0][1] + '\n'
                input = input + 'ASSISTANT Observation: ' + intermediate[1] + '\n'
                if "Status Code: 2" not in intermediate[1]:
                    flag = False
            if 'Could not parse LLM output' in input:
                error_detail['error parser'].append({'api': api['Name'], 'idx': idx, })
                continue
            if "Status Code: 5" in input:
                error_detail['5'].append({'api': api['Name'], 'idx':idx})
                continue
            if "Status Code: 4" in input:
                error_detail['4'].append({'api': api['Name'], 'idx': idx})
                continue
            if flag:
                error_detail['2'].append({'api': api['Name'], 'idx': idx})
                continue
            error_detail['other error'].append({'api': api['Name'], 'idx': idx, })
print(max_steps)
print(len(error_detail['2']))
json.dump(
    error_detail['2'],
    open('./golden_correct_mix.json', "w", encoding="utf-8"),
    indent=4,
    ensure_ascii=False
)

# string. One of: [m, s, f]. [Optional] Specify the unit identifier for temperature and other measurements.
# "I'm feeling a bit bored. Can you cheer me up with a random, useless fact? I'm in the mood for something quirky and unexpected."
# I'm heading out for a run in San Francisco. Can you tell me what the current temperature is and how it feels outside? Also, I'd like to know the humidity and visibility.