"""Text-grad optimizer and prompts."""

from typing import List, Dict
from collections import defaultdict
import logging

from lightrag.optim.optimizer import Optimizer, ParamsT
from lightrag.optim.parameter import Parameter
from lightrag.core.generator import Generator
from lightrag.core import ModelClient, Prompt

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
OPTIMIZER_SYSTEM_PROMPT = """You are part of an optimization system that improves variable to achieve better performance.

You will be asked to creatively and critically improve value of type: {{param_type}}.
You will receive some feedback, and use the feedback to improve the variable.
The feedback may be noisy, identify what is important and what is correct.
Remember:
- Pay attention to the role description of the variable, and the context in which it is used.
- You MUST give your response by sending the improved variable between {{new_variable_start_tag}} $improved_variable {{new_variable_end_tag}} tags.
- Be concise and only modify the variable value in <VARIABLE> tags.
"""


# TGD update instruction # 1. delete ({{variable_desc}})
TGD_PROMPT_TARGET_PARAM = """
<START_OF_VARIABLE_DESC>
Variable type: <TYPE>{{param_type}}</TYPE>
Variable value: <VARIABLE> {{variable_value}} </VARIABLE>
Role Description: <ROLE>{{variable_desc}}</ROLE>.
{% if instruction_to_optimizer %}
Note: {{instruction_to_optimizer}}
{% endif %}
<END_OF_VARIABLE_DESC>

Here are the feedback and context for the variable:
<CONTEXT_FEEDBACK>{{variable_grad}}</CONTEXT_FEEDBACK>
"""


TGD_PROMPT_OUTPUT_FORMAT = """Send the improved variable in the following format:

{{new_variable_start_tag}}the_improved_variable{{new_variable_end_tag}}

Send ONLY the improved variable between the {{new_variable_start_tag}} and {{new_variable_end_tag}} tags, and nothing else.
"""


MOMENTUM_PROMPT_ADDITION = """Here are the past iterations of this variable:

<PAST_ITERATIONS>{{past_values}}</PAST_ITERATIONS>
Similar feedbacks across different steps suggests that the modifications to the variable are insufficient.
If this is the case, please make more significant changes to the variable.
"""


CONSTRAINT_PROMPT_ADDITION = """You must follow the following constraints:
<CONSTRAINTS>{{constraint_text}}</CONSTRAINTS>
"""


IN_CONTEXT_EXAMPLE_PROMPT_ADDITION = """You must base on the following examples when modifying the {{variable_desc}}:

<EXAMPLES>{{in_context_examples}}</EXAMPLES>
"""


TEXT_GRAD_DESC_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>{{optimizer_system_prompt}}<END_OF_SYSTEM_PROMPT><USER>{{user_prompt}}</USER>"""


class TextualGradientDescent(Optimizer):
    __doc__ = """Textual Gradient Descent(LLM) optimizer for text-based variables."""

    proposing: bool = False
    params: ParamsT
    constraints: List[str]

    def __init__(
        self,
        params: ParamsT,
        model_client: ModelClient,
        model_kwargs: Dict[str, object] = {},
        constraints: List[str] = None,
        new_variable_tags: List[str] = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"],
        optimizer_system_prompt: str = OPTIMIZER_SYSTEM_PROMPT,
        in_context_examples: List[str] = None,
        gradient_memory: int = 0,
    ):
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
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples or []
        self.do_in_context_examples = len(self.in_context_examples) > 0
        self.gradient_memory = gradient_memory
        self.gradient_memory_dict = defaultdict(list)
        self.do_gradient_memory = self.gradient_memory > 0
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

    # TODO: optimize with lightrag template for better readability
    def get_gradient_memory_text(self, param: Parameter) -> str:
        grad_memory = ""
        variable_grad_memory = self.gradient_memory_dict[param][-self.gradient_memory :]
        for i, grad_info in enumerate(variable_grad_memory):
            grad_memory += f"\n<FEEDBACK-{i+1}> {grad_info['value']}</FEEDBACK-{i+1}>\n"
        return grad_memory

    def _update_prompt(self, param: Parameter):
        grad_memory = self.get_gradient_memory_text(param)

        # Parameter description
        para_desc_str = Prompt(
            TGD_PROMPT_TARGET_PARAM,
            prompt_kwargs={
                "variable_desc": param.role_desc,
                "variable_value": param.data,
                "variable_grad": param.get_gradient_and_context_text(),
                "param_type": str(param.param_type),
                "instruction_to_optimizer": param.instruction_to_optimizer,
            },
        )()

        tgd_output_format_str = Prompt(
            TGD_PROMPT_OUTPUT_FORMAT,
            prompt_kwargs={
                "new_variable_start_tag": self.new_variable_tags[0],
                "new_variable_end_tag": self.new_variable_tags[1],
            },
        )()

        constraint_str, in_context_example_str, grad_memory_str = None, None, None
        if self.do_constrained:
            constraint_str = Prompt(
                CONSTRAINT_PROMPT_ADDITION,
                prompt_kwargs={"constraint_text": self.constraint_text},
            )()
        if self.do_in_context_examples:
            in_context_example_str = Prompt(
                IN_CONTEXT_EXAMPLE_PROMPT_ADDITION,
                prompt_kwargs={
                    "in_context_examples": "\n".join(self.in_context_examples)
                },
            )()
        if self.do_gradient_memory:
            grad_memory_str = Prompt(
                MOMENTUM_PROMPT_ADDITION,
                prompt_kwargs={"past_values": grad_memory},
            )()

        # Filter out None values before joining
        tgd_user_prompt = "\n".join(
            filter(
                None,
                [
                    para_desc_str,
                    tgd_output_format_str,
                    constraint_str,
                    in_context_example_str,
                    grad_memory_str,
                ],
            )
        )
        log.debug(
            f"Constructed TGD user prompt for {param.role_desc}: {tgd_user_prompt}"
        )
        return tgd_user_prompt

    # TODO: better way to update the gradient memory
    def update_gradient_memory(self, param: Parameter):
        self.gradient_memory_dict[param].append({"value": param.get_gradient_text()})

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
        print("Proposing a new value.")

        for param in self.params:
            if not param.requires_opt:
                log.info(
                    f"Skipping {param.role_desc} as it does not require optimization."
                )
                continue
            print(f"Proposing a new value for {param.alias}.")
            system_prompt = self.optimizer_system_prompt(
                param_type=str(param.param_type)
            )
            user_prompt = self._update_prompt(param)
            prompt_kwargs = {
                "optimizer_system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
            # turn off cache
            response = self.llm_optimizer.call(
                prompt_kwargs=prompt_kwargs, use_cache=not no_cache
            )
            prompt_str = self.llm_optimizer.get_prompt(**prompt_kwargs)
            log.debug(f"TGD LLM optimizer prompt: {prompt_str}")
            proposed_data = response.data
            log.info(f"Response from the optimizer: {response}")
            print(f"Response from the optimizer: {response}")
            # extract the improved variable from the response
            # TODO: make it more robust
            improved_variable = (
                proposed_data.split(self.new_variable_tags[0])[1]
                .split(self.new_variable_tags[1])[0]
                .strip()
            )
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
