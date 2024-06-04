import time
import subprocess
import json
from utils import load_openapi_spec, escape
from agent.agent_prompts import train_prompt_v2, test_prompt_v1, test_prompt_v3
from agent.tools import Tool, GetDetailsTool, tool_projection
from agent.custom_agent import CustomZeroShotAgent, CustomZeroShotAgent2
from transformers import AutoTokenizer
import tiktoken,json
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    # 结构转化，结构不完整则返回0
    try:
        messages = json.loads(messages)
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    except json.JSONDecodeError:
        num_tokens = 0
    return num_tokens

json_list = [
'./generate/single-api/Llama-2-7b-chat-ms_real.json',
             './generate/single-api-static/v3/Llama-2-7b-chat-ms_real_epoch1.json',
'./generate/single-api-static/v3/Llama-2-7b-chat-ms_real_epoch2.json',
'./generate/single-api-static/v3/Llama-2-7b-chat-ms_real_epoch3.json',
'./generate/single-api-dynamic/Llama-2-7b-chat-ms_real_wo_cot_epoch1.json',
'./generate/single-api-dynamic/Llama-2-7b-chat-ms_real_wo_cot_epoch2.json',
'./generate/single-api-dynamic/Llama-2-7b-chat-ms_real_sample_epoch1.json',
'./generate/single-api-dynamic/Llama-2-7b-chat-ms_real_sample_epoch2.json',
'./generate/single-api-dynamic/Llama-2-7b-chat-ms_real_normal_epoch1.json',
'./generate/single-api-dynamic/Llama-2-7b-chat-ms_real_normal_epoch2.json',
'./generate/single-api-dynamic/Llama-2-7b-chat-ms_real_epoch2_normal_epoch1.json',
'./generate/single-api-dynamic/Llama-2-7b-chat-ms_real_epoch2_normal_epoch2.json',

    './generate/single-api/ToolAlpaca-7B_real.json',
             './generate/single-api-static/v3/ToolAlpaca-7B_real_epoch1.json',
'./generate/single-api-static/v3/ToolAlpaca-7B_real_epoch2.json',
'./generate/single-api-dynamic/ToolAlpaca-7B_real_wo_cot_epoch1.json',
'./generate/single-api-dynamic/ToolAlpaca-7B_real_wo_cot_epoch2.json',
'./generate/single-api-dynamic/ToolAlpaca-7B_real_sample_epoch1.json',
'./generate/single-api-dynamic/ToolAlpaca-7B_real_sample_epoch2.json',
'./generate/single-api-dynamic/ToolAlpaca-7B_real_normal_epoch1.json',
'./generate/single-api-dynamic/ToolAlpaca-7B_real_normal_epoch2.json',
'./generate/single-api-dynamic/ToolAlpaca-7B_real_epoch2_normal_epoch1.json',
'./generate/single-api-dynamic/ToolAlpaca-7B_real_epoch2_normal_epoch2.json',

'./generate/single-api/chatgpt-3.5-turbo_real.json',
'./generate/single-api-static/v3/chatgpt-3.5-turbo_real_epoch1.json',
'./generate/single-api-dynamic/chatgpt-3.5-turbo_real_wo_cot_epoch1.json',
'./generate/single-api-dynamic/chatgpt-3.5-turbo_real_wo_cot_epoch2.json',
'./generate/single-api-dynamic/chatgpt-3.5-turbo_real_sample_epoch1.json',
'./generate/single-api-dynamic/chatgpt-3.5-turbo_real_sample_epoch2.json',
'./generate/single-api-dynamic/chatgpt-3.5-turbo_real_normal_epoch1.json',
'./generate/single-api-dynamic/chatgpt-3.5-turbo_real_normal_epoch2.json',
'./generate/single-api-dynamic/chatgpt-3.5-turbo_real_epoch1_normal_epoch1.json',
'./generate/single-api-dynamic/chatgpt-3.5-turbo_real_epoch1_normal_epoch2.json'
             ]

for api_data_path in json_list:
    if 'Llama-2-7b-chat-ms' in api_data_path:
        tokenizer = AutoTokenizer.from_pretrained(r"/home/huan/projects/llm/Llama-2-7b-chat-ms")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif 'ToolAlpaca' in api_data_path:
        tokenizer = AutoTokenizer.from_pretrained(r"/home/huan/projects/llm/ToolAlpaca-7B")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if 'sample' in api_data_path or 'wo_cot' in api_data_path:
        agent_prompt = test_prompt_v3
    else:
        agent_prompt = test_prompt_v1

    if agent_prompt == test_prompt_v1:
        AgentType = CustomZeroShotAgent
    elif agent_prompt == test_prompt_v3:
        AgentType = CustomZeroShotAgent2
    else:
        raise KeyError

    api_datas = json.load(open(api_data_path, "r"))


    total_tokens = 0
    total_count = 0
    for api_idx, api_data in enumerate(api_datas):
        openapi_spec = load_openapi_spec(api_data["Documentation"], replace_refs=True)
        components_descriptions = escape(api_data["Function_Description"]["components"])

        tools = []
        for ext_tool in api_data.get("external_tools", []):
            tools.append(tool_projection[ext_tool]())
        for idx, func_name in enumerate(api_data["Function_Projection"]):
            description = escape(api_data["Function_Description"][func_name])
            if idx == len(api_data["Function_Projection"]) - 1:
                description += components_descriptions
            path, method = api_data["Function_Projection"][func_name]
            tools.append(Tool(
                base_url='http://127.0.0.1:5678' + "/" + api_data["Name"],
                func_name=func_name,
                openapi_spec=openapi_spec,
                path=path,
                method=method,
                description=description,
                retrieval_available="retrieval" in api_data.get("external_tools", [])
            ))
        prompt = AgentType.create_prompt(
            tools,
            prefix=agent_prompt["prefix"],
            suffix=agent_prompt["suffix"],
            format_instructions=agent_prompt["format_instructions"],
            input_variables=["input", "agent_scratchpad"]
        )

        instances = api_data['Instances']
        for ins_idx, instance in enumerate(instances):
            if isinstance(instance, dict):
                for value in instance.values():
                    total_count += 1
                    content = f"{prompt.template}\n".replace('{input}', f"{api_data['Instructions'][ins_idx]}")
                    content = content.replace('{agent_scratchpad}', '\n')
                    if "last_feedbacks" in value.keys():
                        for step in value['last_feedbacks']:
                            if isinstance(step, list):
                                content += f'ASSISTANT Action: {step[0][0]}\nASSISTANT Action Input: {step[0][1]}\n'
                                content += f'ASSISTANT Thought:{step[1]}\n'
                            else:
                                raise KeyError
                    if "dynamic_feedbacks" in value.keys():
                        for step in value['dynamic_feedbacks']:
                            if step is None:
                                continue
                            if isinstance(step, list):
                                content += f'ASSISTANT Action: {step[0][0]}\nASSISTANT Action Input: {step[0][1]}\n'
                                content += f'ASSISTANT Observation:{step[1]}\n'
                            else:
                                raise KeyError
                    if 'intermediate_steps' in value.keys():
                        for step in value['intermediate_steps']:
                            if len(step[0]) == 2:
                                content +=f'ASSISTANT Action: {step[0][0]}\nASSISTANT Action Input: {step[0][1]}\n'
                            elif len(step[0]) == 3:
                                content += f'ASSISTANT Thought:{step[0][2]}\n'
                            else:
                                raise KeyError
                            content += f'ASSISTANT Observation:{step[1]}\n'
                    if 'sample' in api_data_path or 'wo_cot' in api_data_path:
                        content = content.replace('ASSISTANT Thought:', '').replace('ASSISTANT Observation:', 'USER: The response is')


                    if 'gpt-3.5' in api_data_path:
                        total_tokens += num_tokens_from_messages(json.dumps([{"role":"user", "content":content}]), "gpt-3.5-turbo")
                    else:
                        total_tokens += len(tokenizer(content)['input_ids'])

    print(api_data_path, total_tokens, total_count)


