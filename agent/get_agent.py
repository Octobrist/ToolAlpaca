import os
import json
import re
import logging
from typing import List, Tuple, Any, Union

from pydantic import Field
from langchain import LLMChain
from langchain.agents import ZeroShotAgent
from langchain.schema import AgentAction, AgentFinish

from .tools import Tool, GetDetailsTool, tool_projection
from .custom_parser import CustomMRKLOutputParser, CustomGPTOutputParser, CustomGPTMRKLOutputParser
from .custom_agent_executor import CustomAgentExecutor
from utils import load_openapi_spec, escape
from .agent_prompts import train_prompt_v2, test_prompt_v1, test_prompt_v3
from .custom_agent import CustomZeroShotAgent, CustomZeroShotAgent2


logger = logging.getLogger(__name__)


def get_agent(
    llm,
    api_data,
    server_url,
    agent_prompt=train_prompt_v2,
    enable_getDetails=True,
    return_intermediate_steps=True,
    max_iterations = 15
):
        
    openapi_spec = load_openapi_spec(api_data["Documentation"], replace_refs=True)
    components_descriptions = escape(api_data["Function_Description"]["components"])

    tools = [GetDetailsTool()] if not enable_getDetails else []
    for ext_tool in api_data.get("external_tools", []):
        tools.append(tool_projection[ext_tool]())

    for idx, func_name in enumerate(api_data["Function_Projection"]):
        description = escape(api_data["Function_Description"][func_name])
        if idx == len(api_data["Function_Projection"]) - 1:
            description += components_descriptions
        path, method = api_data["Function_Projection"][func_name]
        tools.append(Tool(
            base_url=server_url + "/" + api_data["Name"] if server_url else None,
            func_name=func_name,
            openapi_spec=openapi_spec,
            path=path,
            method=method,
            description=description,
            retrieval_available="retrieval" in api_data.get("external_tools", [])
        ))
    if agent_prompt == test_prompt_v1:
        AgentType = CustomZeroShotAgent
    elif agent_prompt == test_prompt_v3:
        AgentType = CustomZeroShotAgent2
    else:
        AgentType = ZeroShotAgent

    prompt = AgentType.create_prompt(
        tools, 
        prefix=agent_prompt["prefix"], 
        suffix=agent_prompt["suffix"],
        format_instructions=agent_prompt["format_instructions"],
        input_variables=["input", "agent_scratchpad"]
    )

    logger.info(str(prompt))

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # AgentType.return_values = ["output", "Final Thought"]
    AgentType.return_values = ["output"]
    # if max_iterations > 1:

    agent = AgentType(llm_chain=llm_chain, allowed_tools=[t.name for t in tools])
    if agent_prompt != test_prompt_v1:
        agent.output_parser = CustomMRKLOutputParser()

    # if feedback_type.lower() == 'static':
    #     if hasattr(llm, 'model_name') and llm.model_name == 'gpt-3.5-turbo':
    #         agent.output_parser = CustomGPTStaticOutputParser(get_description(api_data['Function_Description']), api_data['Golden_Answers'])
    #     else:
    #         agent.output_parser = CustomStaticOutputParser(get_description(api_data['Function_Description']), api_data['Golden_Answers'])
    # else:
    if hasattr(llm, 'model_name') and llm.model_name == 'gpt-3.5-turbo':
        if agent_prompt != test_prompt_v1:
            agent.output_parser = CustomGPTMRKLOutputParser()
        else:
            agent.output_parser = CustomGPTOutputParser()

    agent_executor = CustomAgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        return_intermediate_steps=return_intermediate_steps,
    )
    return agent_executor

