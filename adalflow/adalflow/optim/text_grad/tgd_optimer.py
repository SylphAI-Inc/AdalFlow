"""Text-grad optimizer and prompts. Also combined methods from ORPO llm optimizer.

With the auto-diff gradients, it made it possible to optimize any prompt parameter in a task pipeline.

https://arxiv.org/abs/2309.03409
Source code: https://github.com/google-deepmind/opro
"""

from typing import List, Dict, TYPE_CHECKING, Optional
from collections import defaultdict
import logging
from dataclasses import field, dataclass


from adalflow.optim.optimizer import TextOptimizer, ParamsT
from adalflow.optim.parameter import Parameter

from adalflow.core.base_data_class import DataClass

if TYPE_CHECKING:
    from adalflow.core import ModelClient


log = logging.getLogger(__name__)

GLOSSARY_TEXT = r"""
### Glossary of tags that will be sent to you:
{# # - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model. #}
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable."""

# customize the system prompt
# prompts, solutions to problems, code, or any other text-based variable. -> to the variable type.
# The optimizer will have an understanding of different variable types.
OPTIMIZER_SYSTEM_PROMPT = r"""You are part of an optimization system that improves variable to achieve better performance.

You will be asked to creatively and critically improve value of type: {{param_type}}.
You will receive some feedback, and use the feedback to improve the variable.
The feedback may be noisy, identify what is important and what is correct.

{# output format #}
You will ONLY output the new variable value in the response between {{new_variable_start_tag}} and {{new_variable_end_tag}} tags.
{# You MUST give your response by sending the improved variable between {{new_variable_start_tag}} $improved_variable {{new_variable_end_tag}} tags.#}

Remember:
- Pay attention to the role description of the variable, and the context in which it is used.
- Be concise and only modify the variable value in <VARIABLE> tags.
- Be creative at generating the new variable value.
"""


# TGD update instruction # 1. delete ({{variable_desc}})
# TGD_PROMPT_TARGET_PARAM = """
# <START_OF_VARIABLE_DESC>
# Variable type: <TYPE>{{param_type}}</TYPE>
# Variable value: <VARIABLE> {{variable_value}} </VARIABLE>
# Role Description: <ROLE>{{variable_desc}}</ROLE>.
# {% if instruction_to_optimizer %}
# Note: {{instruction_to_optimizer}}
# {% endif %}
# <END_OF_VARIABLE_DESC>

# Here are the feedback and context for the variable:
# <CONTEXT_FEEDBACK>{{variable_grad}}</CONTEXT_FEEDBACK>
# """


# TGD_PROMPT_OUTPUT_FORMAT = """Send the improved variable in the following format:

# {{new_variable_start_tag}}the_improved_variable{{new_variable_end_tag}}

# Send ONLY the improved variable between the {{new_variable_start_tag}} and {{new_variable_end_tag}} tags, and nothing else.
# """


# MOMENTUM_PROMPT_ADDITION = """Here are the past iterations of this variable:

# <PAST_ITERATIONS>{{past_values}}</PAST_ITERATIONS>
# Similar feedbacks across different steps suggests that the modifications to the variable are insufficient.
# If this is the case, please make more significant changes to the variable.
# """

# # TODO: add scire fir the previous iterations


# CONSTRAINT_PROMPT_ADDITION = """You must follow the following constraints:
# <CONSTRAINTS>{{constraint_text}}</CONSTRAINTS>
# """


# IN_CONTEXT_EXAMPLE_PROMPT_ADDITION = """You must base on the following examples when modifying the {{variable_desc}}:

# <EXAMPLES>{{in_context_examples}}</EXAMPLES>
# """


TEXT_GRAD_DESC_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{optimizer_system_prompt}}

<END_OF_SYSTEM_PROMPT>

<START_OF_USER>
{#Variable and feedback#}

<START_OF_VARIABLE_DESC>
Variable type: <TYPE>{{param_type}}</TYPE>
Variable value:  {{variable_value}}
Role Description: <ROLE>{{variable_desc}}</ROLE>.
{% if instruction_to_optimizer %}
Note: {{instruction_to_optimizer}}
{% endif %}
<END_OF_VARIABLE_DESC>

Here are the context and feedback for the variable:
<CONTEXT_FEEDBACK>{{variable_grad}}</CONTEXT_FEEDBACK>


{# Momentum #}
{% if past_values %}
Here are the past iterations of this variable:

<PAST_ITERATIONS>{{past_values}}</PAST_ITERATIONS>

Similar feedbacks across different steps suggests that the modifications to the variable are insufficient.
If this is the case, please make more significant changes to the variable.
{% endif %}

{# Constraints #}

{% if constraint_text %}
You must follow the following constraints:
<CONSTRAINTS>{{constraint_text}}</CONSTRAINTS>
{% endif %}

{# In-context examples #}

{% if in_context_examples %}
You must base on the following examples when modifying the {{variable_desc}}:

<EXAMPLES>{{in_context_examples}}</EXAMPLES>
{% endif %}
<END_OF_USER>"""


@dataclass
class Instruction(DataClass):
    __doc__ = "Structure variable values for instructions. Can be used in the history of instructions."
    text: str = field(metadata={"desc": "The instruction text"})
    score: float = field(
        metadata={"desc": "The score of the instruction, range from 0 to 1"}
    )
    responses: Optional[List[str]] = field(
        metadata={"desc": "The responses of using the instruction"}, default=None
    )
    gts: Optional[List[str]] = field(
        metadata={"desc": "The ground truth of the task"}, default=None
    )


from adalflow.tracing.decorators import trace_generator_states


new_variable_tags = ["<NEW_VARIABLE>", "</NEW_VARIABLE>"]


@trace_generator_states()
class TGDOptimizer(TextOptimizer):
    __doc__ = """Textual Gradient Descent(LLM) optimizer for text-based variables."""

    proposing: bool = False
    params: ParamsT
    constraints: List[str]

    def __init__(
        self,
        params: ParamsT,
        model_client: "ModelClient",
        model_kwargs: Dict[str, object] = {},
        constraints: List[str] = None,
        # new_variable_tags: List[str] = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"],
        optimizer_system_prompt: str = OPTIMIZER_SYSTEM_PROMPT,
        in_context_examples: List[str] = None,  # TODO: in-context examples
        num_gradient_memory: int = 0,  # TODO: gradient memory and momentum
    ):
        from adalflow.core.generator import Generator
        from adalflow.core import Prompt

        super().__init__()
        self.params = params
        self.constraints = constraints or []
        self.optimizer_system_prompt = Prompt(
            template=optimizer_system_prompt,
            prompt_kwargs={
                "new_variable_start_tag": new_variable_tags[0],
                "new_variable_end_tag": new_variable_tags[1],
            },
        )
        self.do_constrained = len(self.constraints) > 0
        # self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples or []
        self.do_in_context_examples = len(self.in_context_examples) > 0
        self.num_gradient_memory = num_gradient_memory
        self.gradient_memory_dict = defaultdict(list)  # id to num_gradient_memory
        self.do_gradient_memory = self.num_gradient_memory > 0
        self.llm_optimizer = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=TEXT_GRAD_DESC_TEMPLATE,
        )

    @property
    def constraint_text(self):
        """
        Returns a formatted string representation of the constraints.

        :return: A string containing the constraints in the format "Constraint {index}: {constraint}".
        :rtype: str
        """
        constraints_ordered = [
            f"Constraint {i+1}: {constraint}"
            for i, constraint in enumerate(self.constraints)
        ]
        return "\n".join(constraints_ordered)

    # TODO: optimize with adalflow template for better readability
    def get_gradient_memory_text(self, param: Parameter) -> str:
        grad_memory = ""
        variable_grad_memory = self.gradient_memory_dict[param.id][
            -self.num_gradient_memory :
        ]
        for i, grad_info in enumerate(variable_grad_memory):
            grad_memory += f"\n<FEEDBACK-{i+1}> {grad_info['value']}</FEEDBACK-{i+1}>\n"
        return grad_memory

    def _get_user_prompt_kwargs(self, param: Parameter) -> Dict[str, str]:
        user_prompt_kwargs = {
            "variable_desc": param.role_desc,
            "variable_value": param.data,
            "variable_grad": param.get_gradient_and_context_text(),
            "param_type": str(param.param_type),
            "instruction_to_optimizer": param.instruction_to_optimizer,
            # output format
            # "new_variable_start_tag": self.new_variable_tags[0],
            # "new_variable_end_tag": self.new_variable_tags[1],
            # constraints
            "constraint_text": self.constraint_text if self.do_constrained else None,
            # in-context examples
            "in_context_examples": (
                "\n".join(self.in_context_examples)
                if self.do_in_context_examples
                else None
            ),
            # gradient memory
            "past_values": (
                self.get_gradient_memory_text(param)
                if self.do_gradient_memory
                else None
            ),
        }
        # prompt_str = self.llm_optimizer.get_prompt(**user_prompt_kwargs)
        # print(f"Constructed TGD user prompt for {param.role_desc}: {prompt_str}")
        return user_prompt_kwargs

    # TODO: better way to update the gradient memory
    def update_gradient_memory(self, param: Parameter):
        self.gradient_memory_dict[param.id].append({"value": param.get_gradient_text()})

    def zero_grad(self):
        for p in self.params:
            p.reset_gradients()

    # TODO: in the future can propose multiple values at once
    def propose(self):
        r"""Proposing a value while keeping previous value saved on parameter."""
        if self.proposing:
            raise ValueError("Already proposing a value.")

        # no cache so that new proposal can be made
        no_cache = True
        # print("Proposing a new value.")

        for param in self.params:
            if not param.requires_opt:
                log.info(
                    f"Skipping {param.role_desc} as it does not require optimization."
                )
                continue
            # print(f"Proposing a new value for {param.alias}.")
            system_prompt = self.optimizer_system_prompt(
                param_type=str(param.param_type)
            )
            # user_prompt = self._update_prompt(param)
            user_prompt_kwargs = self._get_user_prompt_kwargs(param)
            prompt_kwargs = {
                "optimizer_system_prompt": system_prompt,
                **user_prompt_kwargs,
            }
            # turn off cache
            response = self.llm_optimizer.call(
                prompt_kwargs=prompt_kwargs, use_cache=not no_cache
            )
            prompt_str = self.llm_optimizer.get_prompt(**prompt_kwargs)
            # print(f"TGD LLM optimizer prompt: {prompt_str}")
            log.debug(f"TGD LLM optimizer prompt: {prompt_str}")
            proposed_data = response.data
            log.info(f"Response from the optimizer: {response}")
            # print(f"Response from the optimizer: {response}")
            # extract the improved variable from the response
            # TODO: make it more robust
            try:
                improved_variable = (
                    proposed_data.split(new_variable_tags[0])[1]
                    .split(new_variable_tags[1])[0]
                    .strip()
                )
            except Exception as e:
                log.error(f"Error extracting improved variable: {e}")
                log.error(f"Proposed data: {proposed_data}")
                improved_variable = proposed_data
            param.propose_data(improved_variable)
        self.proposing = True

    def revert(self):
        """Revert to the previous value when the evaluation is worse."""
        if not self.proposing:
            raise ValueError("Not proposing a value.")
        for param in self.params:
            param.revert_data()
        self.proposing = False

    def step(self):
        """Discard the previous value and keep the proposed value."""
        if not self.proposing:
            raise ValueError("Not proposing a value.")
        for param in self.params:
            param.step_data()
            if self.do_gradient_memory:
                self.update_gradient_memory(param)
        self.proposing = False
