import subprocess
# import openai
# from ast import literal_eval
# import json
#
# def openai_chat_completion(model_engine="gpt-3.5-turbo", in_context=False, temperature = 1):
#     chatLog = []
#     save_ornot = ""
#     if model_engine.startswith("gpt") or model_engine.startswith("text-davinci"):
#         openai.api_base = "https://api.huiyan-ai.cn/v1"
#         openai.api_key = ""
#     elif model_engine.startswith("Wizard") or model_engine.startswith("llama") or model_engine.startswith("vicuna"):
#         openai.api_base = "http://175.6.27.233:8000/v1"
#         openai.api_key = ""
#     else:
#         print(f"[-]Model do not support: {model_engine}")
#         return
#
#     while True:
#         try:
#             user_input = input(">>>User: ")
#             if in_context:
#                 chatLog.append({"role": "user", "content": user_input})
#             else:
#                 chatLog = [{"role": "user", "content": user_input}]
#
#             if model_engine.startswith("text-davinci"):
#                 prompt = "\n".join([f"{i['role']}: {i['content']}" for i in chatLog]) + "\nassistant: "
#                 chat = openai.Completion.create(model = model_engine, prompt = prompt, temperature = temperature)
#             else:
#                 chat = openai.ChatCompletion.create(model = model_engine, messages = chatLog, temperature = temperature)
#             if isinstance(chat, str):
#                 #print(type(chat))
#                 chat = json.loads(chat)
#                 #print(chat)
#                 #nchat = literal_eval(chat)
#                 #print(type(nchat))
#                 #print(nchat)
#             if chat.get("error"):
#                 print("Error: ", chat["error"]['message'])
#                 # chatLog.append({"role": "error", "content": chat["error"]['message']})
#                 break
#             elif chat.get("choices"):
#                 print("here")
#                 answer = chat["choices"][0]['message']['content'].lstrip()
#                 chatLog.append({"role": "assistant", "content": answer})
#                 print(f"\n>>>Assistant:\n{answer}\n{'-'*99}")
#                 print(f"{chat['usage']}\n{'='*99}")
#         except KeyboardInterrupt:
#             print("\n[*]Keyboard Interrupt")
#             save_ornot = input(">>>Save conversation? (y/n): ")
#             break
#         except Exception as e:
#             print(e)
#             if chatLog:
#                 save_ornot = ""
#
# if __name__ == '__main__':
#     openai_chat_completion(model_engine="gpt-3.5-turbo-0125", in_context=False, temperature = 1)


import time
return_code = -1
while return_code != 0:
    # cmd = 'python feedback/response_correct_analsis.py -api ./generate/single-api-static/chatgpt-3.5-turbo_real_epoch1.json -out ./eval/v2/single-api-static --continue'
    # cmd = 'python feedback/response_correct_analsis.py -api ./generate/single-api/chatgpt-3.5-turbo_real.json -out ./eval/v2/ --continue'
    # cmd = 'python feedback/response_correct_analsis.py -api ./generate/single-api/Llama-2-7b-chat-ms_real.json -out ./eval/v3/single-api --continue'
    cmd = 'python feedback/response_correct_analsis.py -api ./generate/single-api-dynamic/Llama-2-7b-chat-ms_real_normal_epoch2.json -out ./eval/v3/single-api-dynamic/ --continue'
    p = subprocess.Popen(cmd, shell=True)
    return_code = p.wait()
    time.sleep(10)

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


# import json
# import time
# from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
#
# from langchain.input import get_color_mapping
# from langchain.tools.base import BaseTool
# from langchain.agents import AgentExecutor
# from langchain.schema import AgentAction, AgentFinish
# from agent.get_agent import get_agent
# from agent.agent_prompts import test_prompt_v1
# from langchain.chat_models import ChatOpenAI
#
# api_data_path = 'generate/single-api/ToolAlpaca-7B_real.json'
# api_data = json.load(open(api_data_path, "r"))
#
# for api_idx, api in enumerate(api_data):
#     agent = get_agent(
#         llm= ChatOpenAI(temperature=0.0),
#         api_data=api,
#         server_url="http://127.0.0.1:1234",
#         agent_prompt=test_prompt_v1,
#         enable_getDetails=True,
#         max_iterations=1
#     )
#     api_name = api['Name']
#     instances = api['Instances']
#     for idx, ins in enumerate(instances):
#         if isinstance(ins, dict):
#             for step in range(len(ins)):
#                 cur_step = ins[str(step)]
#                 if 'intermediate_steps' in cur_step.keys() and len(cur_step['intermediate_steps'])>0:
#                     intermedia_step = cur_step['intermediate_steps'][-1]
#                     if 'Invalid JSON format.' in intermedia_step[1]:
#                         action = AgentAction(intermedia_step[0][0], intermedia_step[0][1], intermedia_step[0][2])
#                         observation = agent.take_action(action)
#                         if 'Invalid JSON format.' not in observation:
#                             api_data[api_idx]['Instances'][idx][str(step)]['intermediate_steps'][-1][1] = observation
#
# with open(api_data_path.replace('_real', '_real_fix'), 'w') as file:
#     json.dump(api_data, file, indent=4)
