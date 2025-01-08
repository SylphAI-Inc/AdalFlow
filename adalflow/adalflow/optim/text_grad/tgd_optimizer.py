"""Text-grad optimizer and prompts. Also combined methods from ORPO llm optimizer.

With the auto-diff gradients, it made it possible to optimize any prompt parameter in a task pipeline.

https://arxiv.org/abs/2309.03409
Source code: https://github.com/google-deepmind/opro
"""

from typing import List, Dict, TYPE_CHECKING, Optional, Any
from collections import defaultdict
import logging
import re
from dataclasses import field, dataclass

from adalflow.optim.optimizer import TextOptimizer, ParamsT
from adalflow.optim.text_grad.backend_engine_prompt import VARIABLE_AND_PEERS_INFO
from adalflow.optim.parameter import Parameter

from adalflow.core.base_data_class import DataClass
from adalflow.tracing.decorators import trace_generator_states
from adalflow.utils.logger import printc
from adalflow.core.types import GeneratorOutput


if TYPE_CHECKING:
    from adalflow.core import ModelClient


log = logging.getLogger(__name__)


@dataclass
class HistoryPrompt(DataClass):
    id: str
    value: str
    eval_score: float


####################################################################################################
# Textual Gradient Descent Optimizer
####################################################################################################
# {% if failed_proposals %}
# Here are the past failed proposals:
# {% for failed_proposal in failed_proposals %}
# {{loop.index}}. {{failed_proposal}}
# {% endfor %}
# {% endif %}

TEXT_GRAD_DESC_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{optimizer_system_prompt}}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER_MESSAGE>
{# Variable and peers info #}
<START_OF_VARIABLE_AND_PEERS_INFO>
{{variable_and_peers_info}}
<END_OF_VARIABLE_AND_PEERS_INFO>
{# system trainable variables #}
{% if system_variables %}
<START_OF_SYSTEM_VARIABLES>
The target variable is used together with these system variables besides of its peers:
{% for system_variable in system_variables %}
{{loop.index}}.
Name: {{system_variable.name}}
Type: {{system_variable.param_type}}
Description: {{system_variable.role_desc}}
WILL_BE_OPTIMIZED: {{system_variable.requires_opt}}
Vaule: {{system_variable.prompt_data}}
{% endfor %}
Strategically plan the role of each system variable to collaborate with each other for final correct answer.
<END_OF_SYSTEM_VARIABLES>
{% endif %}
{# OPRO past history #}
{% if past_history %}
<START_OF_HISTORY_PERFORMANCE>
Here are the best past iterations of this variable along with the validation score.
{% for history in past_history %}
{{loop.index}}. {{history}}
{% endfor %}
IMPORTANT: Your goal is to generate new variable that score higher than all past iterations.
{# Momentum #}
{% if failed_proposals %}
Here are the past failed proposals:
{% for failed_proposal in failed_proposals %}
{{loop.index}}. {{failed_proposal}}
{% endfor %}
{% endif %}
You MUST Try a different approach from the failed proposals.
<END_OF_HISTORY_PERFORMANCE>
{% endif %}
Here are the context and feedback for the variable:
<START_OF_CONTEXT_FEEDBACK>
{{variable_grad}}
<END_OF_CONTEXT_FEEDBACK>
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

# YOU MUST ENSURE the new variable shares the same intent as the original variable.
# You can either rephrase the initial variable, or add more specific instructions based on the feedback.
# You can not change the variable to only fit on one sample if the batch size is larger than 1.

# optimizer system prompt

# Tips:
# 1. Eliminate unnecessary words or phrases.
# 2. Add new elements to address specific feedback.
# 3. Be creative and present the variable differently.
# Provide only the new variable value between {{new_variable_start_tag}} and {{new_variable_end_tag}} tags.
# OPTIMIZER_SYSTEM_PROMPT = r"""
# You are part of an optimization system that refines existing variable based on feedback generated on a batch of input data.

# 1. Address the concerns raised in the feedback while preserving positive aspects.
# 3. Observe past performance patterns when provided and to keep the good quality.
# 4. Consider the variable in the context of its peers if provided.
#    FYI:
#    - If a peer will be optimized itself, do not overlap with its scope.
#    - Otherwise, you can overlap if it is necessary to address the feedback.

# {{output_format_str}}

# YOU MUST ENSURE the new variable shares the same intent as the original variable.
# You can either rephrase the initial variable when no specific feedback found, or add more specific instructions based on the feedback.
# You can not change the variable to only fit on one sample if the batch size is larger than 1.
# {% if instruction_to_optimizer %}
# YOU Should consider user instruction: {{instruction_to_optimizer}}
# {% endif %}
# """

# OPTIMIZER_SYSTEM_PROMPT = r"""
# You are a prompt engineer who excels at refining existing prompts used in LLM in-context learning.

# Your task:
# - **Improve a variable** based on feedback from a batch of input data points.

# ### Context and Requirements

# 1. **Variable Usage**
#    The variable is either an input or output of a functional component. The component schema will be provided.
#    If the same DataID has multiple gradients, it indicates that this component/variable is called repeatedly in a compound system (with a cycle) in the same order that it appears in the gradient list.

# 2. **Key Objectives**
#    1. **Address Feedback**: Resolve concerns raised in the feedback while preserving the positive aspects of the original variable.
#    2. **Peer Awareness**:
#       - If a peer variable will be optimized separately, do not overlap with its scope.
#    3. **Consistency with Past Performance**: Observe patterns from previous iterations and retain beneficial qualities.
#    4. **Be Creative** in your improvements.

# 3. **Additional Notes**
#    - Add new elements to address each specific piece of feedback.
#    - Rephrase or eliminate unnecessary words for clarity.

# {% if instruction_to_optimizer %}
# **User Instructions**: {{instruction_to_optimizer}}
# {% endif %}
# <START_OF_OUTPUT_FORMAT>
# {{output_format_str}}
# <END_OF_OUTPUT_FORMAT>
# """
# <TASK_PIPELINE>
# Here is a summary on the task pipeline you are optimizing:
# retriever: retrieves relevant documents for the question. (Not trainable, you have no control)
# LLM: Answer questions by reading the context  and reason the best answer.
# </TASK_PIPELINE>
OPTIMIZER_SYSTEM_PROMPT = r"""
You are an excellent prompt engineer who works on optimizing a compound LLM system with in-context learning.
Your task is to improve a variable based on feedback from a batch of input data points.

The variable is either input or output of a functional component where the component schema will be provided.
If the same DataID has multiple gradients, it means this component/variable is called multiple times in the compound system(with a cycle) in the same order as it appears in the gradient list.

When the LLM system is complicated with multiple system variables, you need to strategize the role of each
### Your Responsibilities:
1. **Address Feedback**: Resolve concerns raised in the feedback while preserving the positive aspects of the original variable.
2. Observe past performance patterns (when available) to retain good qualities in the variable.
3. **System Awareness**: When other system variables are given, ensure you understand how this variable works in the whole system.
   You have a choice to not update a variable if it is not responsible for the error. Just keep the `update` field as `False`.
4. **Peer Awareness**: This variable works together with Peer variables, ensure you are aware of their roles and constraints.
5. Be Creative. If adding new elements, be concise.

### Your available solutions.
1. Add new elements to address each specific feedback.
2. Add demonstration (e.g., input-reasoning-answer) for tasks that require strong reasoning skills.
3. Rephrase(for more clarity) to address the feedback.
4. You can also eliminate unnecessary words to improve clarity.

### prompt engineering practices:
1. Set Context and Role: Establish a specific identity or domain expertise for the AI to guide style, knowledge, and constraints.
2. Demonstration: Construct input-reasoning-answer example especially for tasks that require strong reasoning skills.
3. Be Specific and Clear: Clearly define instructions, desired format, and constraints to ensure accurate and relevant outputs.
4. Leverage Constraints and Formatting: Explicitly direct how the answer should be structured (e.g., bullet points, tables, or tone).
5. Self-Consistency / Verification Prompts: Prompt the model to check its own logic for errors, inconsistencies, or missing details.

{{output_format_str}}

{% if instruction_to_optimizer %}
**Additional User Instructions**: {{instruction_to_optimizer}}
{% endif %}
"""

# <TASK_PIPELINE>
# Here is a summary on the task pipeline you are optimizing:
# query_generator(a trainable LLM): "generates a sub-query based on the initial query"
# retriever: "retrieves relevant documents based on the sub-query"
# llm(a trainable LLM): "Answer a question with available context with exact answer extracted from the context"
# duplicator: "functional part to depulicate the documents, no trainable part, no need to have feedback or to be optimized."

# The query_generator+ retriever is called twice in the pipeline as the question requires two sub-queries.
# And the retrieved documents are deduplicated and combined to form the final context.
# The final context is then passed to the llm to generate the answer where we want to use the exact phrase from the context.
# </TASK_PIPELINE>
# OPTIMIZER_SYSTEM_PROMPT = r"""
# You are a prompt engineer exels at refining existing prompts used in LLM in-context learning.
# Your task is to improve a variable based on feedback from a batch of input data points.

# The variable is either input or output of a functional component where the component schema will be provided.
# If the same DataID has multiple gradients, it means this component/variable is called multiple times in the compound system(with a cycle) in the same order as it appears in the gradient list.

# ### YOU MUST ENSURE:
# 1. **Address Feedback**: Resolve concerns raised in the feedback while preserving the positive aspects of the original variable.
# 2. **Peer Awareness**:
#    - If a peer will be optimized itself, do not overlap with its scope.
# 3. Observe past performance patterns (when available) to retain good qualities in the variable.
# 4. Be Creative.
# 5. The new variable MUST have better performance than all previous iterations.

# ### NOTES:
# 1. Add new elements to address each specific feedback.
# 2. rephrase to address the feedback.
# 3. You can also eliminate unnecessary words to improve clarity.

# ### Common prompt engineering practices:
# 1. Set Context and Role: Establish a specific identity or domain expertise for the AI to guide style, knowledge, and constraints.
# 2. Zero-Shot vs. Few-Shot Prompting: Decide whether to provide examples (few-shot) or none (zero-shot) to shape responses and format.
# 3. Be Specific and Clear: Clearly define instructions, desired format, and constraints to ensure accurate and relevant outputs.
# 4. Leverage Constraints and Formatting: Explicitly direct how the answer should be structured (e.g., bullet points, tables, or tone).
# 5. Self-Consistency / Verification Prompts: Prompt the model to check its own logic for errors, inconsistencies, or missing details.

# {% if instruction_to_optimizer %}
# **More Instructions**: {{instruction_to_optimizer}}
# {% endif %}

# {{output_format_str}}
# """

# When no feedback is provided(high batch performance), you rephrase the variable.
### Tips:
# 1. Patterns like "think step by step" helps the model reason better. You should try to maintain the chain of thought.


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


@dataclass
class TGDData(DataClass):
    reasoning: str = field(
        metadata={
            "desc": "Which solution did you choose, which prompt engineering technique did you use? Why? Be Concise (maximum 2 sentences)"
        }
    )
    proposed_variable: str = field(
        metadata={"desc": "The proposed variable"}, default=None
    )
    update: bool = field(
        default=True,
        metadata={
            "desc": "Depending on the feedback, update the variable if it is responsible for the error, else, keep it"
        },
    )


@dataclass
class TGDOptimizerTrace(DataClass):
    api_kwargs: Dict[str, Any] = field(
        metadata={
            "desc": "The api_kwargs for components like Generator and Retriever that pass to the model client"
        },
        default=None,
    )
    output: TGDData = field(
        metadata={"desc": "The output of the TGD optimizer"}, default=None
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


@trace_generator_states()
class TGDOptimizer(TextOptimizer):
    __doc__ = """Textual Gradient Descent(LLM) optimizer for text-based variables."""

    proposing: bool = False
    params: ParamsT
    constraints: List[str]
    params_history: Dict[str, List[HistoryPrompt]] = {}  # id to history
    failed_proposals: Dict[str, List[HistoryPrompt]] = {}  # only need the value

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
        max_failed_proposals: int = 2,
    ):
        from adalflow.core.generator import Generator
        from adalflow.core import Prompt
        from adalflow.components.output_parsers.dataclass_parser import DataClassParser

        super().__init__()
        self.params = params
        self.constraints = constraints or []
        self.data_class = TGDData
        self.output_parser = DataClassParser(
            data_class=self.data_class, return_data_class=True, format_type="json"
        )
        self.optimizer_system_prompt = Prompt(
            template=optimizer_system_prompt,
            prompt_kwargs={
                # "new_variable_start_tag": new_variable_tags[0],
                # "new_variable_end_tag": new_variable_tags[1],
                "output_format_str": """Your output should be formatted as a standard JSON instance with the following schema:
```
{
    "reasoning": "Why the variable is proposed this way (str) (required)",
    "proposed_variable": "The proposed variable (str) (required)"
}
```"""
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
            output_processors=self.output_parser,
        )

        self.max_past_history = max_past_history
        self.max_failed_proposals = max_failed_proposals

        # initate the past history for each parameter
        for param in self.params:
            self.params_history[param.id] = []
            self.failed_proposals[param.id] = []

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

    def add_failed_proposal(self):
        """Save a copy of the current value of the parameter in the failed proposals."""
        for param in self.params:
            failed_proposal = HistoryPrompt(
                id=param.id,
                value=param.data,
                eval_score=None,
            )
            self.failed_proposals[param.id].append(failed_proposal)
            if len(self.failed_proposals[param.id]) > self.max_failed_proposals:
                for _ in range(
                    len(self.failed_proposals[param.id]) - self.max_failed_proposals
                ):
                    self.failed_proposals[param.id].pop()
        # if param_id not in self.failed_proposals:
        #     self.failed_proposals[param_id] = []
        # failed_proposal = HistoryPrompt(
        #     id=param_id,
        #     value=value,
        #     eval_score=None,
        # )
        # self.failed_proposals[param_id].append(failed_proposal)
        # if len(self.failed_proposals[param_id]) > self.max_failed_proposals:
        #     for _ in range(len(self.failed_proposals[param_id]) - self.max_failed_proposals):
        #         self.failed_proposals[param_id].pop()

    def render_failed_proposals(self, param_id: str) -> List[str]:
        if param_id not in self.failed_proposals:
            return []
        return [
            history.to_yaml(exclude=["id", "eval_score"])
            for history in self.failed_proposals[param_id]
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

        system_params = [
            p.get_param_info()
            for p in self.params
            if p.id != param.id and p not in param.peers
        ]
        peers_params = [p.get_param_info() for p in param.peers]
        variable_and_peer_info = self.variable_and_peers_info.call(
            variable=param.get_param_info(), peers=peers_params
        )

        variable_grad = param.get_gradients_component_schema(skip_correct_sample=False)

        user_prompt_kwargs = {
            "variable_and_peers_info": variable_and_peer_info,
            "variable_grad": variable_grad,  # param.get_gradient_and_context_text(
            #   skip_correct_sample=False
            # ),
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
            # failed proposals
            "failed_proposals": (
                self.render_failed_proposals(param.id)
                if self.max_failed_proposals
                else None
            ),
            "system_variables": system_params,
        }

        return user_prompt_kwargs

    # TODO: better way to update the gradient memory
    def update_gradient_memory(self, param: Parameter):
        self.gradient_memory_dict[param.id].append(
            {"value": param.get_gradient_and_context_text(skip_correct_sample=True)}
        )

    def zero_grad(self):
        for p in self.params:
            p.reset_gradients()

    # TODO: in the future can propose multiple values at once
    def propose(self):
        r"""Proposing a value while keeping previous value saved on parameter."""
        if self.proposing:
            raise ValueError("Already proposing a value.")

        printc("Proposing a new value.", color="magenta")

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
            try:
                response: GeneratorOutput = self.llm_optimizer.call(
                    prompt_kwargs=prompt_kwargs, use_cache=not no_cache
                )
            except Exception as e:
                printc(f"Error in the optimizer: {e}", color="red")
                raise e
            if not isinstance(response, GeneratorOutput):
                raise TypeError(f"Wrong response type: {type(response)}")

            prompt_str = self.llm_optimizer.get_prompt(**prompt_kwargs)
            log.debug(f"TGD LLM optimizer prompt: {prompt_str}")
            printc(f"TGD LLM optimizer prompt:: {prompt_str}", color="blue")
            proposed_data: TGDData = (
                response.data
                if response.data is not None
                else TGDData(
                    reasoning="No reasoning",
                    proposed_variable=response.raw_response,
                    update=False,
                )
            )
            printc(f"Response from the optimizer: {response}", color="blue")

            log.info(f"Response from the optimizer: {response}")
            if not proposed_data.update:
                printc(f"No update is required for {param.name}", color="yellow")
                param.propose_data(param.data)
            else:
                improved_variable = proposed_data.proposed_variable
                param.propose_data(improved_variable)
            param.trace_optimizer(api_kwargs=prompt_str, response=response)
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
            param.trace_optimizer(api_kwargs=None, response=None)
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
    # print(response)
