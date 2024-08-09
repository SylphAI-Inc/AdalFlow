r"""Based and optimized from ORPO llm optimizer.

https://arxiv.org/abs/2309.03409
Source code: https://github.com/google-deepmind/opro
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import field, dataclass

from adalflow.core.base_data_class import DataClass

if TYPE_CHECKING:

    from adalflow.core.model_client import ModelClient


from adalflow.optim.optimizer import Optimizer, ParamsT

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

{# Current instruction #}
{% if current_instruction %}
<CURRENT_INS>
Here is the current instruction:
{{current_instruction}}
</CURRENT_INS>
{% endif %}

{# Training samples on current instruction #}
{% if training_samples %}
<TRAINING_SAMPLES>
Here are some training samples on the current instruction:
{% for sample in training_samples %}
- {{sample}}
{% endfor %}
</TRAINING_SAMPLES>
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
"""


@dataclass
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


# TODO: adapt this to multiple parameters who requires_opt
# TODO: combine batch errors few shot
# TODO: combine few-shot with llm optimizer
# TODO: support multi processors and multiple proposals at the same time
class LLMOptimizer(Optimizer):
    __doc__ = r"""Default LLM optimizer for task instruction.

    User should always provide a starter instruction which provides the source of truth for the task.
    The optimizer will generate new instructions that can score higher than all previous instructions.
    """

    def __init__(
        self,
        params: ParamsT,
        model_client: "ModelClient",
        model_kwargs: Dict[str, Any],
    ):
        r"""Initialize the generator with the model client and the model kwargs."""
        super().__init__()
        from adalflow.core.generator import Generator

        self.instruction_parameter = list(params)[
            0
        ]  # for now only support one parameter
        # Ensure the temperature is at least 1
        model_kwargs["temperature"] = max(1, model_kwargs.get("temperature", 1))

        self.instruction_history: List[Instruction] = (
            []
        )  # trace the history of the instructions
        self.starter_instruction: Optional[str] = None
        if self.instruction_parameter.data is not None:
            self.starter_instruction = self.instruction_parameter.data
        # self.proposed: Optional[str] = None
        # self.current: Optional[str] = None
        self.prompt_kwargs = {
            "starter_instruction": self.starter_instruction,
            "instructions": self.instruction_history,
        }
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=LLM_OPTIMIZER_TEMPLATE,
            prompt_kwargs=self.prompt_kwargs,
        )
        self.proposing = False

    def reset(self):
        self.instruction_history = []

    def zero_grad(self):
        pass

    def propose(self, training_samples: List[str] = None):
        r"""Propose a new instruction using the generator."""
        from adalflow.core.generator import GeneratorOutput

        if self.proposing:
            raise ValueError("Already proposing a new instruction.")
        max_run: int = 5
        current_instruction = self.instruction_parameter.data
        additional_prompt_kwargs = {
            "current_instruction": current_instruction,
            "training_samples": training_samples,
        }
        for _ in range(max_run):
            instruction: GeneratorOutput = self.generator(
                prompt_kwargs=additional_prompt_kwargs
            )
            self.generator.print_prompt(**additional_prompt_kwargs)
            if instruction.data is not None:
                instruction = instruction.data
                break

        # self.proposed = instruction
        self.instruction_parameter.propose_data(instruction)
        self.proposing = True

    # Step
    def step(self, score: float):
        r"""Load the proposed instruction to the current instruction to complete the optimization step."""
        if not self.proposing:
            raise ValueError("No proposed instruction to step.")

        self.instruction_parameter.step_data()
        data = self.instruction_parameter.data
        self.instruction_history.append(Instruction(text=data, score=score))
        self.proposing = False

    # revert
    def revert(self):
        r"""The proposed is not leading to a better instruction, reset the parameter."""
        if not self.proposing:
            raise ValueError("No proposed instruction to revert.")
        # self.proposed = None
        self.instruction_parameter.revert_data()
        self.proposing = False
