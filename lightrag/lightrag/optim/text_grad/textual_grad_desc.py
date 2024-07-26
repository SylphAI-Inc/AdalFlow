from typing import List, Dict
from collections import defaultdict
import logging

from lightrag.optim.optimizer import Optimizer, ParamsT
from lightrag.optim.parameter import Parameter
from lightrag.core.generator import Generator
from lightrag.core import ModelClient, Prompt

log = logging.getLogger(__name__)

GLOSSARY_TEXT = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable."""

# System prompt to TGD
# OPTIMIZER_SYSTEM_PROMPT = (
#     "You are part of an optimization system that improves text (i.e., variable). "
#     "You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable. "
#     "You will receive some feedback, and use the feedback to improve the variable. "
#     "The feedback may be noisy, identify what is important and what is correct. "
#     "Pay attention to the role description of the variable, and the context in which it is used. "
#     "This is very important: You MUST give your response by sending the improved variable between {new_variable_start_tag} {{improved variable}} {new_variable_end_tag} tags. "
#     "The text you send between the tags will directly replace the variable.\n\n"
#     f"{GLOSSARY_TEXT}"
# )
# NOTE: {{improved variable}} is used to escape the curly braces in the string, after jinja2 templating, it will be {improved variable}
# {{{}}} is to replace it with the actual value
# {{{{}}}} is to have the actual variable in the string
OPTIMIZER_SYSTEM_PROMPT = (
    """You are part of an optimization system that improves text (i.e., variable).

You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable.
You will receive some feedback, and use the feedback to improve the variable.
The feedback may be noisy, identify what is important and what is correct.
Remember:
- Pay attention to the role description of the variable, and the context in which it is used.
- You MUST give your response by sending the improved variable between {{new_variable_start_tag}} $improved_variable {{new_variable_end_tag}} tags.
"""
    + GLOSSARY_TEXT
)


print(OPTIMIZER_SYSTEM_PROMPT)

# TGD update instruction
TGD_PROMPT_PREFIX = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Here is the context and feedback we got for the variable:\n\n"
    "<CONTEXT>{variable_grad}</CONTEXT>\n\n"
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)


TGD_PROMPT_SUFFIX = (
    "Send the improved variable "
    "in the following format:\n\n{new_variable_start_tag}the_improved_variable{new_variable_end_tag}\n\n"
    "Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else."
)

MOMENTUM_PROMPT_ADDITION = (
    "Here are the past iterations of this variable:\n\n"
    "<PAST_ITERATIONS>{past_values}</PAST_ITERATIONS>\n\n"
    "Similar feedbacks across different steps suggests that the modifications to the variable are insufficient."
    "If this is the case, please make more significant changes to the variable.\n\n"
)

CONSTRAINT_PROMPT_ADDITION = (
    "You must follow the following constraints:\n\n"
    "<CONSTRAINTS>{constraint_text}</CONSTRAINTS>\n\n"
)

IN_CONTEXT_EXAMPLE_PROMPT_ADDITION = (
    "You must base on the following examples when modifying the {variable_desc}:\n\n"
    "<EXAMPLES>{in_context_examples}</EXAMPLES>\n\n"
)


def construct_tgd_prompt(
    do_momentum: bool = False,
    do_constrained: bool = False,
    do_in_context_examples: bool = False,
    **optimizer_kwargs,
):
    """
    Construct the textual gradient descent prompt.

    :param do_momentum: Whether to include momentum in the prompt.
    :type do_momentum: bool, optional
    :param do_constrained: Whether to include constraints in the prompt.
    :type do_constrained: bool, optional
    :param do_in_context_examples: Whether to include in-context examples in the prompt.
    :type do_in_context_examples: bool, optional
    :param optimizer_kwargs: Additional keyword arguments for formatting the prompt. These will be things like the variable description, gradient, past values, constraints, and in-context examples.
    :return: The TGD update prompt.
    :rtype: str
    """

    prompt = TGD_PROMPT_PREFIX.format(**optimizer_kwargs)

    if do_momentum:
        prompt += MOMENTUM_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_constrained:
        prompt += CONSTRAINT_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_in_context_examples:
        prompt += IN_CONTEXT_EXAMPLE_PROMPT_ADDITION.format(**optimizer_kwargs)

    prompt += TGD_PROMPT_SUFFIX.format(**optimizer_kwargs)

    return prompt


class TextualGradientDescent(Optimizer):
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
        r"""Initialize the optimizer."""
        # super().__init__(params)
        self.params = params
        self.constraints = constraints or []
        self.optimizer_system_prompt = Prompt(
            template=optimizer_system_prompt,
            prompt_kwargs={
                "new_variable_start_tag": new_variable_tags[0],
                "new_variable_end_tag": new_variable_tags[1],
            },
        )()
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
            template="""<SYS>{{optimizer_system_prompt}}</SYS><USER>{{user_prompt}}</USER> Your response:""",
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
        optimizer_information = {
            "variable_desc": param.role_desc,
            "variable_value": param.data,
            "variable_grad": param.get_gradient_and_context_text(),
            "variable_short": param.get_short_value(),
            "constraint_text": self.constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(
                self.in_context_examples
            ),  # TODO: use lightrag in_context_examples
            "gradient_memory": grad_memory,
        }

        prompt = construct_tgd_prompt(
            do_constrained=self.do_constrained,
            do_in_context_examples=(
                self.do_in_context_examples and (len(self.in_context_examples) > 0)
            ),
            do_gradient_memory=(self.do_gradient_memory and (grad_memory != "")),
            **optimizer_information,
        )

        log.info("TextualGradientDescent prompt for update", extra={"prompt": prompt})
        return prompt

    # TODO: better way to update the gradient memory
    def update_gradient_memory(self, param: Parameter):
        self.gradient_memory_dict[param].append({"value": param.get_gradient_text()})

    def zero_grad(self):
        for p in self.params:
            p.reset_gradients()

    def step(self):
        r"""Take a step in the optimization process.
        It will update the parameters with the new values."""
        for param in self.params:
            user_prompt = self._update_prompt(param)
            response = self.llm_optimizer.call(
                prompt_kwargs={
                    "optimizer_system_prompt": self.optimizer_system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            response_data = response.data
            log.info(f"Response from the optimizer: {response}")
            # extract the improved variable from the response
            # TODO: make it more robust
            improved_variable = (
                response_data.split(self.new_variable_tags[0])[1]
                .split(self.new_variable_tags[1])[0]
                .strip()
            )
            param.update_value(improved_variable)
            if self.do_gradient_memory:
                self.update_gradient_memory(param)
