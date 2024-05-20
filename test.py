# import time
# return_code = -1
# while return_code != 0:
#     # cmd = 'python feedback/response_correct_analsis.py -api ./generate/single-api-static/chatgpt-3.5-turbo_real_epoch1.json -out ./eval/v2/single-api-static --continue'
#     # cmd = 'python feedback/response_correct_analsis.py -api ./generate/single-api/chatgpt-3.5-turbo_real.json -out ./eval/v2/ --continue'
#     # cmd = 'python feedback/response_correct_analsis.py -api ./generate/single-api/Llama-2-7b-chat-ms_real.json -out ./eval/v3/single-api --continue'
#     cmd = 'python feedback/response_correct_analsis.py -api ./generate/single-api-dynamic/ToolAlpaca-7B_real_sample_epoch1.json -out ./eval/v3/single-api-dynamic/ --continue'
#     p = subprocess.Popen(cmd, shell=True)
#     return_code = p.wait()
#     time.sleep(10)

# cmds = [
# 'python instance_generation/static_generation.py -inp ./generate/single-api-static/v3/Llama-2-7b-chat-ms_real_epoch1.json -out ./generate -llm /home/huan/projects/llm/Llama-2-7b-chat-ms --agent_prompt test_v1 --real --max_iterations 1 --epoch 2 --server_url http://127.0.0.1:1234',
# 'python instance_generation/static_generation.py -inp ./generate/single-api-static/Llama-2-7b-chat-ms_real_epoch2.json -out ./generate -llm /home/huan/projects/llm/Llama-2-7b-chat-ms --agent_prompt test_v1 --real --max_iterations 1 --epoch 3 --server_url http://127.0.0.1:1234',
# 'python instance_generation/static_generation.py -inp ./generate/single-api-static/Llama-2-7b-chat-ms_real_epoch3.json -out ./generate -llm /home/huan/projects/llm/Llama-2-7b-chat-ms --agent_prompt test_v1 --real --max_iterations 1 --epoch 4 --server_url http://127.0.0.1:1234',
# 'python instance_generation/static_generation.py -inp ./generate/single-api-static/Llama-2-7b-chat-ms_real_epoch4.json -out ./generate -llm /home/huan/projects/llm/Llama-2-7b-chat-ms --agent_prompt test_v1 --real --max_iterations 1 --epoch 5 --server_url http://127.0.0.1:1234',
#     # 'python instance_generation/static_generation.py -inp ./generate/single-api-static/v3/ToolAlpaca-7B_real_epoch2.json -out ./generate -llm /home/huan/projects/llm/ToolAlpaca-7B --agent_prompt test_v1 --real --max_iterations 1 --epoch 3 --server_url http://127.0.0.1:1234'
# ]
# for cmd in cmds:
#     p = subprocess.Popen(cmd, shell=True)
#     return_code = p.wait()


import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain.input import get_color_mapping
from langchain.tools.base import BaseTool
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish
from agent.get_agent import get_agent
from agent.agent_prompts import test_prompt_v1
from langchain.chat_models import ChatOpenAI
eval_mix = json.load(open('./data/eval_mix.json', "r"))
# api_data_path = 'eval/golden-eval_real.json'
api_data_path = 'eval/golden-eval_simulated.json'
api_data = json.load(open(api_data_path, "r"))
error_detail = {'2':[], '4':[], '5':[], 'json':[], 'error parser':[], 'other error':[]}
count = 0
for api_idx, api in enumerate(api_data):
    api_name = api['Name']
    instances = api['Instances']
    for idx, ins in enumerate(instances):
        if isinstance(ins, dict):
            if "intermediate_steps" not in ins.keys():
                error_detail['other error'].append({'api': api['Name'], 'idx': idx, 'error': ins['error']})
                continue
            steps = ins["intermediate_steps"]
            eval_mix[11+api_idx]['Golden_Answers'][idx] = [{'Action':step[0][0], 'Action_Input':step[0][1]} for step in steps]
            input = ins['input'] + '\n'
            flag = True
            break_flag = False
            for step in steps:
                if 'Could not parse LLM output' in step[1]:
                    error_detail['error parser'].append({'api': api['Name'], 'idx': idx, 'steps':len(steps)})
                    break_flag = True
                    break
                elif 'Invalid JSON format.' in step[1]:
                    error_detail['json'].append({'api': api['Name'], 'idx': idx, 'steps':len(steps)})
                    break_flag = True
                    break
                else:
                    if 'Status Code: 200' not in step[1]:
                         flag = False
                    input = input + 'ASSISTANT Action: ' + step[0][0] + '\n'
                    input = input + 'ASSISTANT Action Input: ' + step[0][1] + '\n'
                    input = input + 'ASSISTANT Observation: ' + step[1] + '\n'

            if not break_flag and "Status Code: 5" in input:
                error_detail['5'].append(
                    {'api': api['Name'], 'idx': idx, 'steps':len(steps)})
            elif not break_flag and "Status Code: 4" in input:
                error_detail['4'].append(
                    {'api': api['Name'], 'idx': idx, 'steps':len(steps)})
            elif flag:
                error_detail['2'].append(
                    {'api': api['Name'], 'idx': idx, 'steps':len(steps)})
                if len(steps) > 1:
                    count += 1
        else:
            print(1)
            # for step in range(len(ins)):
            #     cur_step = ins[str(step)]
            #     if 'cur_step' in cur_step.keys():
            #         api_data[api_idx]['Instances'][idx][str(step)]["dynamic_feedbacks"] = [cur_step['cur_step']]
            #         api_data[api_idx]['Instances'][idx][str(step)].pop('cur_step')
print(count)
print(error_detail)

json.dump(
    eval_mix,
    open('./data/eval_mix.json', "w", encoding="utf-8"),
    indent=4,
    ensure_ascii=False
)

# for api_idx, api in enumerate(api_data):
#     agent = get_agent(
#         llm= ChatOpenAI(temperature=0.0),
#         api_data=api,
#         server_url="http://127.0.0.1:5678",
#         agent_prompt=test_prompt_v1,
#         enable_getDetails=not False,
#         max_iterations=1
#     )
#     api_name = api['Name']
#     instances = api['Instances']
#     for idx, ins in enumerate(instances):
#         if isinstance(ins, dict):
#             if "intermediate_steps" not in ins.keys():
#                 continue
#             if {'api': api_name, 'idx':idx} not in error_detail['4']:
#                 continue
#             steps = ins["intermediate_steps"]
#             for step_idx, step in enumerate(steps):
#                 action = AgentAction(step[0][0], step[0][1],
#                                      f'\nASSISTANT Action: {step[0][0]}\nASSISTANT Action Input: {step[0][1]}')
#                 cur_oberservation = agent.take_action(action)
#                 api_data[api_idx]['Instances'][idx]["intermediate_steps"][step_idx][1] = cur_oberservation
#
# json.dump(
#     api_data,
#     open('eval/golden-eval_simulated3.json', "w", encoding="utf-8"),
#     indent=4,
#     ensure_ascii=False
# )