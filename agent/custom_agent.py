from langchain.schema import AgentAction, AgentFinish
from pydantic import Field
from langchain.agents import ZeroShotAgent
from langchain.agents.agent import AgentOutputParser
from typing import List, Tuple, Any, Union, Dict

from .custom_parser import CustomMRKLOutputParser2, CustomMRKLOutputParser


class CustomZeroShotAgent(ZeroShotAgent):
    output_parser: AgentOutputParser = Field(default_factory=CustomMRKLOutputParser2)

    @classmethod
    def _get_default_output_parser(cls, **kwargs) -> AgentOutputParser:
        return CustomMRKLOutputParser2()

    def get_full_inputs_with_cur_step(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)
        if 'dynamic_feedbacks' in kwargs.keys() and kwargs['dynamic_feedbacks'] is not None :
            for feedback in kwargs['dynamic_feedbacks']:
                thoughts += f'\nASSISTANT Action: {feedback[0][0]}\nASSISTANT Action Input: {feedback[0][1]}\nASSISTANT Observation: {feedback[1]}\n' \
                            f'ASSISTANT Thought: I think I have completed the user\'s question\n' \
                            f'USER: No, I think your actions and action inputs do not meet my expectations. Please regenerate them.'
        if 'cur_step' in kwargs.keys():
            cur_step = kwargs['cur_step']
            if 'dynamic' in kwargs.keys():
                thoughts += f'\nASSISTANT Action: {cur_step[0][0]}\nASSISTANT Action Input: {cur_step[0][1]}\n' \
                            f'ASSISTANT Observation: {cur_step[1]}\nASSISTANT Thought: I think I have completed the user\'s question\n' \
                            f'USER: No, I think your actions and action inputs do not meet my expectations. ' \
                            f'The question is: {kwargs["input"]} I won\'t give any more information. You should change the input and retry or call another function, don\'t ask me any questions, and regenerate it right now.\n' \
                            # f'ASSISTANT Thought: I will regenerate a new action and a new action input.\n ASSISTANT Action: '

            else:
                thoughts += f'\nASSISTANT Action: {cur_step[0][0]}\nASSISTANT Action Input: {cur_step[0][1]}\nASSISTANT Observation:\nASSISTANT Thought: {cur_step[1]}'
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs
    def plan( # for gpt
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs_with_cur_step(intermediate_steps, **kwargs)
        full_output = self.llm_chain.predict(**full_inputs)
        return self.output_parser.parse(full_output)

    @classmethod
    def _get_default_output_parser(cls, **kwargs) -> AgentOutputParser:
        return CustomMRKLOutputParser2()

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "ASSISTANT Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "ASSISTANT Thought:"


class CustomZeroShotAgent2(ZeroShotAgent): # for sample
    output_parser: AgentOutputParser = Field(default_factory=CustomMRKLOutputParser2)

    @classmethod
    def _get_default_output_parser(cls, **kwargs) -> AgentOutputParser:
        return CustomMRKLOutputParser()

    def get_full_inputs_with_cur_step(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)
        if 'dynamic_feedbacks' in kwargs.keys() and kwargs['dynamic_feedbacks'] is not None :
            for feedback in kwargs['dynamic_feedbacks']:
                thoughts += f'\nASSISTANT Action: {feedback[0][0]}\nASSISTANT Action Input: {feedback[0][1]}\n' \
                            f'USER: {feedback[1]} I think your actions and action inputs do not meet my expectations. Please regenerate them.'

        if 'cur_step' in kwargs.keys() and kwargs['cur_step'] is not None:
            cur_step = kwargs['cur_step']
            thoughts += f'\nASSISTANT Action: {cur_step[0][0]}\nASSISTANT Action Input: {cur_step[0][1]}\n' \
                        f'USER: {cur_step[1]} I think your actions and action inputs do not meet my expectations. Please regenerate them.' \

        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    def plan( # for gpt
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs_with_cur_step(intermediate_steps, **kwargs)
        full_output = self.llm_chain.predict(**full_inputs)
        return self.output_parser.parse(full_output)

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "ASSISTANT Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "ASSISTANT Thought:"
