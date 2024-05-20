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
from utils import load_openapi_spec, analyze_openapi_spec, escape
from agent.tools import Tool, GetDetailsTool, tool_projection
import torch
import deepspeed

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-api", "--api_data_path", type=str, required=True)
parser.add_argument("-out", "--output_dir", type=str, default="")

parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5678")
parser.add_argument("--output_prefix", type=str, default="api_data")
parser.add_argument("--agent_prompt", type=str, default="train_v2")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--length", type=int, default=-1)
parser.add_argument("--use_cache", action="store_true", default=False)
parser.add_argument("--real", action="store_true", default=False)
parser.add_argument("--without_getDetails", action="store_true", default=False)
args = parser.parse_args()


api_data = json.load(open(args.api_data_path, "r"))

if args.length == -1:
    args.length = len(api_data) - args.offset

api_data = api_data[args.offset: args.offset + args.length]
final_output_path = os.path.join(args.output_dir,
                                 f"golden-{args.api_data_path.split('/')[-1]}")

if args.use_cache:
    res = requests.get(f"{args.server_url}/__simulator_cache__/open")

for api_idx, api in tqdm(enumerate(api_data)):
    api["Instances"] = []
    if "Instructions" not in api or len(api["Instructions"]) == 0:
        continue
    openapi_spec = load_openapi_spec(api["Documentation"])
    input_valid, output_valid = analyze_openapi_spec(openapi_spec)
    if input_valid and output_valid:

        openapi_spec = load_openapi_spec(api["Documentation"], replace_refs=True)
        components_descriptions = escape(api["Function_Description"]["components"])

        tools = [GetDetailsTool()]
        for ext_tool in api.get("external_tools", []):
            tools.append(tool_projection[ext_tool]())

        for idx, func_name in enumerate(api["Function_Projection"]):
            description = escape(api["Function_Description"][func_name])
            if idx == len(api["Function_Projection"]) - 1:
                description += components_descriptions
            path, method = api["Function_Projection"][func_name]
            tools.append(Tool(
                base_url=args.server_url + "/" + api["Name"] if args.server_url else None,
                func_name=func_name,
                openapi_spec=openapi_spec,
                path=path,
                method=method,
                description=description,
                retrieval_available="retrieval" in api.get("external_tools", [])
            ))
        name_to_tool_map = {}
        for tool in tools:
            name_to_tool_map[tool.name] = tool
        Answers = []
        for idx, gas in enumerate(api["Golden_Answers"]):
            inst = api["Instructions"][idx]
            output = {}
            try:
                if len(api.get("Authentication", [])) > 0:
                    inst += "\nAuthentication information: " + \
                            " ".join([f"{k}={v}" for k, v in api["Authentication"].items()])
                output['input'] = inst
                output["intermediate_steps"] = []
                for ga in gas:
                    tool = name_to_tool_map[ga['Action']]
                    observation = tool.run(
                        ga['Action_Input'],
                    )
                    output["intermediate_steps"].append([[ga['Action'], ga['Action_Input']], observation])
                    json.dumps(output, ensure_ascii=4)
            except json.JSONDecodeError:
                output = str(output)
            except Exception as e:
                logger.error(e)
                output = {"error": str(e)}

            if args.use_cache:
                res = requests.get(f"{args.server_url}/__simulator_cache__/clear/{api['Name']}")
                print(res.text)

            Answers.append(output)

        api["Instances"] = Answers
        json.dump(
            api_data,
            open(final_output_path, "w", encoding="utf-8"),
            indent=4,
            ensure_ascii=False
        )

json.dump(
    api_data,
    open(final_output_path, "w", encoding="utf-8"),
    indent=4,
    ensure_ascii=False
)
