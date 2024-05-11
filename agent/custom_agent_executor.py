import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain.input import get_color_mapping
from langchain.tools.base import BaseTool
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish


from .tools import CustomInvalidTool

class CustomAgentExecutor(AgentExecutor):
    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        if 'intermediate_steps' in inputs.keys():
            for intermedia_step in inputs['intermediate_steps']:
                intermediate_steps.append((AgentAction(intermedia_step[0][0], intermedia_step[0][1], f'\nASSISTANT Action: {intermedia_step[0][0]}\nASSISTANT Action Input: {intermedia_step[0][1]}'), intermedia_step[1]))
            inputs.pop('intermediate_steps')
        # if 'cur_step' in inputs.keys():
        #     cur_step = inputs['cur_step']
        #     intermediate_steps.append((AgentAction(cur_step[0][0], cur_step[0][1], f'\nASSISTANT Action: {cur_step[0][0]}\nASSISTANT Action Input: {cur_step[0][1]}\n ASSISTANT Observation:\n ASSISTANT Thought: {cur_step[1]}'), ''))
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map, color_mapping, inputs, intermediate_steps
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(next_step_output, intermediate_steps)

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(tool_return, intermediate_steps)
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps)

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        # Call the LLM to see what to do.
        output = self.agent.plan(intermediate_steps, **inputs)
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            self.callback_manager.on_agent_action(
                agent_action, verbose=self.verbose, color="green"
            )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                # =============================== modify ===============================
                # give GetDetailsTool more kwargs
                tool_run_kwargs["inputs"] = inputs
                # =============================== modify ===============================
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = CustomInvalidTool().run(
                    agent_action.tool,
                    all_tools = list(name_to_tool_map.keys()),
                    verbose=self.verbose,
                    color=None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result

    def take_action(
        self,
        agent_action,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        self.callback_manager.on_agent_action(
            agent_action, verbose=self.verbose, color="green"
        )
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            observation = tool.run(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = CustomInvalidTool().run(
                agent_action.tool,
                all_tools = list(name_to_tool_map.keys()),
                verbose=self.verbose,
                color=None,
                **tool_run_kwargs,
            )
        return observation