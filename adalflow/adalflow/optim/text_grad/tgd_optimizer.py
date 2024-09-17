"""Text-grad optimizer and prompts. Also combined methods from ORPO llm optimizer.

With the auto-diff gradients, it made it possible to optimize any prompt parameter in a task pipeline.

https://arxiv.org/abs/2309.03409
Source code: https://github.com/google-deepmind/opro
"""

from typing import List, Dict, TYPE_CHECKING, Optional
from collections import defaultdict
import logging
import re
from dataclasses import field, dataclass


from adalflow.optim.optimizer import TextOptimizer, ParamsT
from adalflow.optim.text_grad.backend_engine_prompt import VARIABLE_AND_PEERS_INFO
from adalflow.optim.parameter import Parameter

from adalflow.core.base_data_class import DataClass

if TYPE_CHECKING:
    from adalflow.core import ModelClient


log = logging.getLogger(__name__)


# Tips:
# 1. Eliminate unnecessary words or phrases.
# 2. Add new elements to address specific feedback.
# 3. Be creative and present the variable differently.
OPTIMIZER_SYSTEM_PROMPT = r"""
You are part of an optimization system that refines existing variable values based on feedback.

Your task: Propose a new variable value in response to the feedback.
1. Address the concerns raised in the feedback while preserving positive aspects.
2. Observe past performance patterns when provided and to keep the good quality.
3. Consider the variable in the context of its peers if provided.
   FYI:
   - If a peer will be optimized itself, do not overlap with its scope.
   - Otherwise, you can overlap if it is necessary to address the feedback.

Output:
Provide only the new variable value between {{new_variable_start_tag}} and {{new_variable_end_tag}} tags.

Tips:
1. Eliminate unnecessary words or phrases.
2. Add new elements to address specific feedback.
3. Be creative and present the variable differently.
{% if instruction_to_optimizer %}
4. {{instruction_to_optimizer}}
{% endif %}
"""


@dataclass
class HistoryPrompt(DataClass):
    id: str
    value: str
    eval_score: float


TEXT_GRAD_DESC_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{optimizer_system_prompt}}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER_MESSAGE>
{#Variable and feedback#}
{{variable_and_peers_info}}
{# ORPO past history #}
{% if past_history %}
<START_OF_HISTORY_PERFORMANCE>
Here are the past iterations of this variable along with the validation score.
{% for history in past_history %}
{{loop.index}}. {{history}}
{% endfor %}
IMPORTANT: Your goal is to generate new variable values that score higher than all previous iterations.
<END_OF_HISTORY_PERFORMANCE>
{% endif %}
Here are the context and feedback for the variable:
<START_OF_CONTEXT_FEEDBACK>
{{variable_grad}}
<END_OF_CONTEXT_FEEDBACK>
{# Momentum #}
{% if past_values %}
Here are the past iterations of this variable:
<PAST_ITERATIONS>
{{past_values}}
</PAST_ITERATIONS>
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
<END_OF_USER_MESSAGE>
"""


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


new_variable_tags = ["<VARIABLE>", "</VARIABLE>"]


def extract_new_variable(text: str) -> str:
    pattern = re.compile(r"<VARIABLE>(.*?)</VARIABLE>", re.DOTALL)

    # Find all matches
    matches = pattern.findall(text)

    if len(matches) == 0:
        return text.strip()
    log.debug(f"Extracted new variable: {matches[0].strip()}")
    return matches[0].strip()


class TGDOptimizer(TextOptimizer):
    __doc__ = """Textual Gradient Descent(LLM) optimizer for text-based variables."""

    proposing: bool = False
    params: ParamsT
    constraints: List[str]
    params_history: Dict[str, List[HistoryPrompt]] = {}  # id to history

    def __init__(
        self,
        params: ParamsT,
        model_client: "ModelClient",
        model_kwargs: Dict[str, object] = {},
        constraints: List[str] = None,
        # new_variable_tags: List[str] = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"],
        optimizer_system_prompt: str = OPTIMIZER_SYSTEM_PROMPT,
        in_context_examples: List[str] = None,  # TODO: in-context examples
        num_gradient_memory: int = 0,  # TODO: gradient memory and momentum, for now it is not useful
        max_past_history: int = 3,
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
        self.variable_and_peers_info = Prompt(
            template=VARIABLE_AND_PEERS_INFO,
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

        self.max_past_history = max_past_history

        # initate the past history for each parameter
        for param in self.params:
            self.params_history[param.id] = []

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

    def add_score_to_params(self, val_score: float):
        for param in self.params:
            self.add_score_to_current_param(param.id, param, val_score)

    def add_score_to_current_param(self, param_id: str, param: Parameter, score: float):
        if param_id not in self.params_history:
            raise ValueError(f"Parameter {param_id} not found in the history.")
        if param.id != param_id:
            raise ValueError(
                f"Parameter {param_id} does not match the target parameter."
            )

        history = HistoryPrompt(
            id=param_id,
            value=str(param.data),
            eval_score=score,
        )
        self.add_history(param_id, history)

    def add_history(self, param_id: str, history: HistoryPrompt):
        if param_id not in self.params_history:
            self.params_history[param_id] = []
        self.params_history[param_id].append(history)
        # sort score from the highest to the lowest
        self.params_history[param_id] = sorted(
            self.params_history[param_id], key=lambda x: x.eval_score, reverse=True
        )
        # delete the lowest score if it exceeds the max_past
        if len(self.params_history[param_id]) > self.max_past_history:
            for _ in range(len(self.params_history[param_id]) - self.max_past_history):
                self.params_history[param_id].pop()

    def render_history(self, param_id: str) -> List[str]:
        if param_id not in self.params_history:
            return []
        return [
            history.to_yaml(exclude=["id"]) for history in self.params_history[param_id]
        ]

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

        variable_and_peer_info = self.variable_and_peers_info.call(
            variable=param.get_param_info(), peers=param.peers  # param.peers
        )

        user_prompt_kwargs = {
            "variable_and_peers_info": variable_and_peer_info,
            "variable_grad": param.get_gradient_and_context_text(),
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
            # past history
            "past_history": (
                self.render_history(param.id) if self.max_past_history else None
            ),
        }

        return user_prompt_kwargs

    # TODO: better way to update the gradient memory
    def update_gradient_memory(self, param: Parameter):
        self.gradient_memory_dict[param.id].append(
            {"value": param.get_gradient_and_context_text()}
        )

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

            # print(f"Proposing a new value for {param.name}.")
            system_prompt = self.optimizer_system_prompt(
                param_type=str(param.param_type),
                instruction_to_optimizer=param.instruction_to_optimizer,
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
            log.debug(f"TGD LLM optimizer prompt: {prompt_str}")
            proposed_data = response.data
            log.info(f"Response from the optimizer: {response}")
            # extract the improved variable from the response
            # TODO: make it more robust
            improved_variable = extract_new_variable(proposed_data)
            param.propose_data(improved_variable)
            if self.do_gradient_memory:
                self.update_gradient_memory(param)
        self.proposing = True

    def revert(self):
        """Revert to the previous value when the evaluation is worse."""
        if not self.proposing:
            raise ValueError("Not proposing a value.")
        for param in self.params:
            if not param.requires_opt:
                continue
            param.revert_data()
        self.proposing = False

    def step(self):
        """Discard the previous value and keep the proposed value."""
        if not self.proposing:
            raise ValueError("Not proposing a value.")
        for param in self.params:
            if not param.requires_opt:
                continue
            param.step_data()

        self.proposing = False


if __name__ == "__main__":
    # test the prompt history
    data = {
        "id": "1",
        "value": "test",
        "eval_score": 0.5,
    }
    history = HistoryPrompt.from_dict(data)

    print(history)
    history_yaml = history.to_yaml()
    print(history_yaml)

    template = r"""<START_OF_SYSTEM_PROMPT>
    {% if past_history %}
    Here are the past iterations of this variable along with the validation score.
    {% for history in past_history %}
    {{loop.index}}. {{history}}
    {% endfor %}
    {% endif %}
    <END_OF_SYSTEM_PROMPT>"""

    import adalflow as adal

    prompt = adal.Prompt(template=template)

    histories = [history_yaml]
    prompt_kwargs = {
        "past_history": histories,
    }
    response = prompt(**prompt_kwargs)
    print(response)
