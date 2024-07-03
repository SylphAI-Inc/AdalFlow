r"""
Based and optimized from ORPO llm optimizer.

https://arxiv.org/abs/2309.03409
Source code: https://github.com/google-deepmind/opro
"""

from typing import Dict, Any, List, Optional
from dataclasses import field
from copy import deepcopy

from lightrag.core.base_data_class import DataClass

from lightrag.core.model_client import ModelClient
from lightrag.core.generator import Generator, GeneratorOutput
from lightrag.core.parameter import Parameter
from lightrag.optim.optimizer import Optimizer

# TODO: add the responses and gts
LLM_OPTIMIZER_TEMPLATE = r"""<SYS>
Your task is to generate an new instruction that can score higher than all previous instructions.
{# starter instruction #}
{%if starter_instruction %}
<STARTER_INS>
Here is the starter instruction:
{{starter_instruction}}
</STARTER_INS>
{% endif %}

{# history #}
{% if instructions %}
<HISTORY_INS>
Below are some of your previous instructions and their scores, the higher the score the better the instruction:
{% for instruction in instructions %}
- {{loop.index}}.
- text: {{instruction.text}}
- score: {{instruction.score}})
{% if instruction.responses is defined %}
- responses: {{instruction.responses}}
{% endif %}
{% if instruction.gts is defined %}
- gts: {{instruction.gts}}
{% endif %}
____
{% endfor %}
</HISTORY_INS>
{% endif %}

{# More task specification #}
- Your new instruction should be different from all previous instructions.
- It should be clear, concise, and effective.
- Do not change core information provided in the starter instruction.
</SYS>
New Instruction:
"""


# @dataclass
class Instruction(DataClass):
    # prefix will be the same as text
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
    # for classification, use the named labels instead of label index


# TODO: combine few-shot with llm optimizer
# TODO: support multi processors and multiple proposals at the same time
class LLMOptimizer(Optimizer):
    __doc__ = r"""Default LLM optimizer for task instruction.

    User should always provide a starter instruction which provides the source of truth for the task.
    The optimizer will generate new instructions that can score higher than all previous instructions.
    """

    def __init__(
        self,
        parameter: Parameter,
        model_client: ModelClient,
        model_kwargs: Dict[str, Any],
    ):
        r"""Initialize the generator with the model client and the model kwargs."""
        super().__init__()
        self.instruction_parameter = parameter
        # Ensure the temperature is at least 1
        model_kwargs["temperature"] = max(1, model_kwargs.get("temperature", 1))
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=LLM_OPTIMIZER_TEMPLATE,
        )
        self.instruction_history: List[Instruction] = []
        self.starter_instruction: Optional[str] = None
        if self.instruction_parameter.data is not None:
            self.starter_instruction = self.instruction_parameter.data
        self.proposed: Optional[str] = None
        self.current: Optional[str] = None
        self.prompt_kwargs = {
            "starter_instruction": self.starter_instruction,
            "instructions": self.instruction_history,
        }

    def reset(self):
        self.instruction_history = []

    def propose(self):
        r"""Propose a new instruction using the generator."""
        max_run: int = 5
        for _ in range(max_run):
            instruction: GeneratorOutput = self.generator(
                prompt_kwargs=self.prompt_kwargs
            )
            if instruction.data is not None:
                instruction = instruction.data
                break

        self.proposed = instruction
        self.instruction_parameter.update_value(instruction)

    def update_parameter(self, score: float):
        r"""Load the proposed instruction to the current instruction to complete the optimization step."""
        self.current = deepcopy(self.proposed)
        self.proposed = None
        if self.current is not None:
            self.instruction_history.append(Instruction(text=self.current, score=score))
            self.instruction_parameter.update_value(self.current)

    def reset_parameter(self):
        r"""The proposed is not leading to a better instruction, reset the parameter."""
        self.proposed = None
        self.instruction_parameter.update_value(self.current)
