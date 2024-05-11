import re
from typing import Union, NamedTuple

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

FINAL_ANSWER_ACTION = "Final Answer:"

class AgentStop(NamedTuple):
    error_type: str
    details: str

class CustomMRKLOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {
                    "output": text.split(FINAL_ANSWER_ACTION)[-1].strip(),
                    "Final Thought": text.rsplit(FINAL_ANSWER_ACTION, 1)[0].strip(),
                 }, text
            )
        # \s matches against tab/newline/whitespace
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(action, action_input.strip(" ").strip('"'), text)

class CustomMRKLOutputParser2(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        final_answer_action = "ASSISTANT Response:"
        if final_answer_action in text:
            return AgentFinish(
                {
                    "output": text,
                    # "Final Thought": text.rsplit(final_answer_action, 1)[0].strip(),
                 }, text
            )
        # \s matches against tab/newline/whitespace
        regex = r"ASSISTANT\s*Action\s*\d*\s*:(.*?)\nASSISTANT\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            return AgentFinish(
                {
                    "output": f"Could not parse LLM output: `{text}`",
                    # "Final Thought": ""
                }, text
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(action, action_input.strip(" ").strip('"'), text)

# huan
class CustomGPTOutputParser(AgentOutputParser):
    def parse(self, ori_text: str) -> Union[AgentAction, AgentFinish]:
        tmp_text = ori_text
        for stop_word in ['\nASSISTANT Observation:', '\n\tASSISTANT Observation:']:
            if stop_word in tmp_text:
                tmp_text = tmp_text.split(stop_word)[0]
                break
        final_answer_action = "ASSISTANT Response:"
        if final_answer_action in tmp_text:
            return AgentFinish(
                {
                    "output": tmp_text,
                    # "Final Thought": tmp_text.rsplit(final_answer_action, 1)[0].strip(),
                 }, tmp_text
            )
        # \s matches against tab/newline/whitespace
        regex = r"ASSISTANT\s*Action\s*\d*\s*:(.*?)\nASSISTANT\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, tmp_text, re.DOTALL)
        if not match:
            return AgentFinish(
                {
                    "output": f"Could not parse LLM output: `{ori_text}`",
                    # "Final Thought": ""
                }, ori_text
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(action, action_input.strip(" ").strip('"'), tmp_text)

class CustomStaticOutputParser: # undo
    def __init__(self, desc, golden):
        self.desc = desc
        self.golden = golden

    def parse(self, text: str, golden: int) -> Union[AgentAction, AgentFinish, AgentStop]:
        final_answer_action = "ASSISTANT Response:"
        if final_answer_action in text:
            return AgentFinish(
                {
                    "output": text.split(final_answer_action)[-1].strip(),
                    # "Final Thought": text.rsplit(final_answer_action, 1)[0].strip(),
                 }, text
            )
        # \s matches against tab/newline/whitespace
        regex = r"ASSISTANT\s*Action\s*\d*\s*:(.*?)\nASSISTANT\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match: # NO API CALL
            raise ValueError(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # todo: 静态分析
        return AgentAction(action, action_input.strip(" ").strip('"'), text)

class CustomGPTStaticOutputParser: # undo
    def __init__(self, desc, golden):
        self.desc = desc
        self.golden = golden

    def parse(self, text: str, golden: int) -> Union[AgentAction, AgentFinish, AgentStop]:
        final_answer_action = "ASSISTANT Response:"
        if final_answer_action in text:
            return AgentFinish(
                {
                    "output": text.split(final_answer_action)[-1].strip(),
                    # "Final Thought": text.rsplit(final_answer_action, 1)[0].strip(),
                 }, text
            )
        # \s matches against tab/newline/whitespace
        regex = r"ASSISTANT\s*Action\s*\d*\s*:(.*?)\nASSISTANT\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match: # NO API CALL
            raise ValueError(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # todo: 静态分析
        return AgentAction(action, action_input.strip(" ").strip('"'), text)