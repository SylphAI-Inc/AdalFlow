"""ReAct agent implementation using the base agent."""

from typing import List, Union, Callable, Optional, Any, Dict
import logging
from adalflow.components.output_parsers import JsonOutputParser

from adalflow.core.generator import Generator
from adalflow.core.func_tool import FunctionTool, AsyncCallable
from adalflow.core.tool_manager import ToolManager
from adalflow.core.types import Function
from adalflow.optim.parameter import Parameter, ParameterType
from adalflow.core.model_client import ModelClient

from .base_agent import (
    BaseAgent,
    BasePlanner,
    BaseToolManager,
    BaseMemory,
    Step,
    AgentOutput,
)

log = logging.getLogger(__name__)
react_agent_task_desc = r"""
<START_OF_TASK_SPEC>
You are an excellent task planner.
Answer the input query using the tools provided below with maximum accuracy.

Each step you will read the previous thought, Action(name, kwargs), and Observation(execution result of the action) and then provide the next Thought and Action.

Follow function docstring to best call the tool.
- For simple queries: Directly call the ``finish`` action and provide the answer.
- For complex queries:
    - Step 1: Read the user query and divide it into multisteps. Start with the first tool/subquery.
    - Call one tool at a time to solve each subquery/subquestion. \
    - At step 'finish', give the final answer based on all previous steps.
REMEMBER:
- Action MUST call one of the tools. It CANNOT be empty.
- You will ALWAYS END WITH 'finish' tool to finish the task directly with answer or failure message.
- When the tool is a class method and when class_instance exists, use <class_instance_value>.<func_name> to call instead (NOT the CLASS NAME)
<END_OF_TASK_SPEC>
"""

DEFAULT_REACT_AGENT_SYSTEM_PROMPT = r"""<START_OF_SYSTEM_PROMPT>
{{react_agent_task_desc}}
- You cant use more than {{max_steps}} steps. At the {{max_steps}}th current step, must finish with answer.

{# Tools #}
{% if tools %}
<START_OF_TOOLS>
Tools and instructions:
{% for tool in tools %}
{{ loop.index }}.
{{tool}}
------------------------
{% endfor %}
<END_OF_TOOLS>
{% endif %}
{# Context Variables #}
{% if context_variables is not none %}
<START_OF_CONTEXT>
You have access to context_variables with the following keys:
{% for key, value in context_variables.items() %}
{{ key }}
------------------------
{% endfor %}
You can either pass context_variables or context_variables['key'] to the tools depending on the tool's requirements.
<END_OF_CONTEXT>
{% endif %}
{# output format and examples for output format #}
<START_OF_OUTPUT_FORMAT>
{{output_format_str}}
<END_OF_OUTPUT_FORMAT>
{% if examples %}
<START_OF_EXAMPLES>
Examples:
{% for example in examples %}
{{example}}
------------------------
{% endfor %}
<END_OF_EXAMPLES>
{% endif %}
<END_OF_SYSTEM_PROMPT>
-----------------
<START_OF_USER_QUERY>
Input query:
{{ input_str }}
_____________________
Current Step/Max Step: {{step_history|length + 1}} / {{max_steps}}
{# Step History #}
{% if step_history %}
<STEPS>
Your previous steps:
{% for history in step_history %}
Step {{ loop.index }}.
{% if history.action %}
"thought": "{{history.action.thought}}",
"name": "{{history.action.name}},
"kwargs": {{history.action.kwargs}}",
{% endif %}
"Observation": "{{history.observation}}"
------------------------
{% endfor %}
</STEPS>
{% endif %}
<END_OF_USER_QUERY>
"""


class ReActPlanner(BasePlanner):
    """ReAct-specific planner implementation."""

    def __init__(
        self,
        model_client: ModelClient,
        model_kwargs: Dict = {},
        template: Optional[str] = None,
        examples: List[Function] = [],
    ):
        super().__init__(model_client, model_kwargs)
        self.template = template or DEFAULT_REACT_AGENT_SYSTEM_PROMPT
        self.examples = examples

        # Initialize output parser with strict JSON requirements
        self.output_parser = JsonOutputParser(
            data_class=Function,
            examples=self.examples,
            return_data_class=True,
            include_fields=["thought", "name", "kwargs"],
        )

        # Initialize generator with proper output format
        self.generator = Generator(
            template=self.template,
            model_client=model_client,
            model_kwargs=model_kwargs,
            output_processors=self.output_parser,
        )

    def plan(self, input: str, context: Dict) -> Function:
        """Plan the next action using ReAct format."""
        print("Reaching for fun ****************")
        print("CONTEXT_VARIABLES", context)
        prompt_kwargs = {
            "input_str": input,
            "step_history": context.get("step_history", []),
            "tools": context.get("tools", []),
            "max_steps": context.get("max_steps", 10),
            "context_variables": context.get("context_variables", {}),
            "react_agent_task_desc": Parameter(
                name="react_agent_task_desc",
                data=react_agent_task_desc,
                # data="You are an excellent task planner. Answer the input query using the tools provided below with maximum accuracy.\n\nEach step you will read the previous thought, Action(name, kwargs), and Observation(execution result of the action) and then provide the next Thought and Action.\n\n<START_OF_TASK_SPEC>\nFollow function docstring to best call the tool.\n- For simple queries: Directly call the 'finish' action and answer with a concise 'yes' or 'no' when it fits.\n- For complex queries:\n    - Step 1: Understand the main subject(s) and context of the user query accurately.\n    - Step 2: Break down the query into multisteps, starting with the first tool/subquery.\n    - Ensure each step accurately reflects the subjects under consideration.\n    - Continuously verify your extracted information and logic for factual accuracy using concise comparisons.\n    - At step 'finish', conclude with a precise final answer.\nREMEMBER:\n- Action MUST call one of the tools. It CANNOT be empty.\n- You will ALWAYS END WITH 'finish' tool to conclude the task directly with an answer or failure message.\n- When the tool is a class method and when class_instance exists, use <class_instance_value>.<func_name> to call instead (NOT the CLASS NAME).\n<END_OF_TASK_SPEC>",
                role_desc="Task instruction for the agent to plan steps to solve a question in sequential and multi-steps to get the final answer. \
                For optimizer: you need to adapt this to the current specific task.",
                param_type=ParameterType.PROMPT,
                requires_opt=True,
            ),
            "examples": Parameter(
                name="examples",
                data=self.examples,
                role_desc="Examples for the ReAct agent.",
                param_type=ParameterType.DEMOS,
                requires_opt=True,
            ),
            "output_format_str": self.output_parser.format_instructions(),
        }

        try:
            response = self.generator(prompt_kwargs=prompt_kwargs)
            if not response or not response.data:
                raise ValueError("No valid response generated")

            return response.data

        except Exception as e:
            log.error(f"Error generating plan: {str(e)}")
            # Provide a fallback response
            return Function(
                thought="Failed to generate plan, falling back to finish",
                name="finish",
                kwargs={"answer": f"Sorry, I encountered an error: {str(e)}"},
            )


class ReActToolManager(BaseToolManager):
    """ReAct-specific tool manager implementation."""

    def __init__(
        self,
        tools: List[Union[Callable, AsyncCallable, FunctionTool]],
        add_llm_as_fallback: bool = True,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Dict = {},
    ):
        super().__init__(tools)
        self.add_llm_as_fallback = add_llm_as_fallback

        # Add default finish tool if not already present
        self._add_default_finish_tool()

        # Add LLM fallback if requested
        if add_llm_as_fallback and model_client:
            self._add_llm_fallback(model_client, model_kwargs)

        # Initialize tool manager
        self.tool_manager = ToolManager(tools=self.tools)

    def _add_default_finish_tool(self):
        """Add a default finish tool if one doesn't already exist."""
        # Check if a finish tool already exists
        finish_exists = any(
            (
                tool.name == "finish"
                if isinstance(tool, FunctionTool)
                else getattr(tool, "__name__", "") == "finish"
            )
            for tool in self.tools
        )

        if not finish_exists:

            def finish(answer: str, **kwargs) -> str:
                """Finish the conversation with a final answer."""
                return answer

            self.tools.append(FunctionTool(finish))
            log.info("Added default 'finish' tool to ReActAgent")

    def _add_llm_fallback(self, model_client: ModelClient, model_kwargs: Dict):
        """Add LLM as a fallback tool."""
        llm_tool = Generator(model_client=model_client, model_kwargs=model_kwargs)

        def llm_fallback(input: str, **kwargs) -> str:
            """Fallback tool that uses LLM to answer queries."""
            try:
                output = llm_tool(prompt_kwargs={"input_str": input})
                return output.data if output else None
            except Exception as e:
                log.error(f"Error using llm_fallback: {e}")
                return None

        self.tools.append(FunctionTool(llm_fallback))

    def execute(self, action: Function) -> Any:
        """Execute an action using the tool manager."""
        if not action or not action.name:
            raise ValueError("Invalid action")

        result = self.tool_manager(expr_or_fun=action, step="execute")

        if not result:
            raise ValueError(f"Failed to execute action: {action}")

        return result.output


class ReActMemory(BaseMemory):
    """ReAct-specific memory implementation."""

    def retrieve(self, query: str) -> List[Step]:
        """Retrieve relevant steps based on query."""
        # Simple implementation - can be enhanced with vector search etc.
        return [step for step in self.steps if query.lower() in str(step).lower()]


class ReActAgent(BaseAgent):
    """ReAct agent implementation using the base agent.

    The agent automatically adds a default 'finish' tool that can be used to complete
    the conversation with a final answer. Users do not need to manually add this tool.
    """

    def __init__(
        self,
        tools: List[Union[Callable, AsyncCallable, FunctionTool]],
        max_steps: int = 10,
        add_llm_as_fallback: bool = True,
        model_client: ModelClient = None,
        model_kwargs: Dict = {},
        template: Optional[str] = None,
        examples: List[Function] = [],
        context_variables: Optional[Dict] = None,
        use_cache: bool = True,
        debug: bool = False,
    ):
        # Initialize components
        planner = ReActPlanner(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            examples=examples,
        )

        tool_manager = ReActToolManager(
            tools=tools,
            add_llm_as_fallback=add_llm_as_fallback,
            model_client=model_client,
            model_kwargs=model_kwargs,
        )

        memory = ReActMemory()

        super().__init__(
            planner=planner,
            tool_manager=tool_manager,
            memory=memory,
            max_steps=max_steps,
            context_variables=context_variables,
            use_cache=use_cache,
            debug=debug,
        )

    def _should_stop(self, step: Step) -> bool:
        """Check if we should stop based on ReAct rules."""
        if not step.action:
            return True

        return step.action.name == "finish"

    def _get_answer(self, step_history: List[Step]) -> Any:
        """Get the final answer from step history."""
        if not step_history:
            return None

        last_step = step_history[-1]
        return last_step.observation

    def train_step(self, input: str, target: Any) -> Dict:
        """Training step implementation."""
        self.train()
        output = self.call(input)
        loss = self._compute_loss(output, target)
        return {"loss": loss}

    def eval_step(self, input: str) -> AgentOutput:
        """Evaluation step implementation."""
        self.eval()
        return self.call(input)

    def _compute_loss(self, output: AgentOutput, target: Any) -> float:
        """Compute loss for training."""
        # Implement loss computation based on your needs
        raise NotImplementedError
