"""Text-grad optimizer and prompts. Also combined methods from ORPO llm optimizer.

With the auto-diff gradients, it made it possible to optimize any prompt parameter in a task pipeline.

https://arxiv.org/abs/2309.03409
Source code: https://github.com/google-deepmind/opro
"""

from typing import List, Dict, TYPE_CHECKING, Optional, Any
import logging
import re
import numpy as np
from dataclasses import field, dataclass

from adalflow.optim.optimizer import TextOptimizer, ParamsT
from adalflow.optim.text_grad.backend_engine_prompt import VARIABLE_AND_PEERS_INFO
from adalflow.optim.parameter import Parameter
from adalflow.core import DataComponent

from adalflow.core.base_data_class import DataClass
from adalflow.core.types import GeneratorOutput
import xml.etree.ElementTree as ET


if TYPE_CHECKING:
    from adalflow.core import ModelClient


log = logging.getLogger(__name__)


@dataclass
class HistoryPrompt(DataClass):
    id: str
    value: str
    eval_score: float
    method: str = field(default=None)
    reasoning: str = field(default=None)


####################################################################################################
# Textual Gradient Descent Optimizer
####################################################################################################
# TODO: make the uesr message to task instruction

TEXT_GRAD_DESC_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{optimizer_system_prompt}}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER_MESSAGE>

<OPTIMIZATION_CONTEXT>
Current optimization iteration: {{steps}} steps since your last improvement.
Optimization strategy: Use more aggressive updates after 3 iterations without improvement.
</OPTIMIZATION_CONTEXT>

<TARGET_CONTENT_TO_OPTIMIZE>
{# Variable and peers info #}
{{variable_and_peers_info}}
</TARGET_CONTENT_TO_OPTIMIZE>
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
Value: {{system_variable.prompt_data}}
{% endfor %}
Strategically plan the role of each system variable to collaborate with each other for final correct answer.
<END_OF_SYSTEM_VARIABLES>
{% endif %}
{# OPRO past history #}
{% if past_history %}
<START_OF_HISTORY_PERFORMANCE>
Here are the best past iterations.
{% for history in past_history %}
{{loop.index}}. {{history}}
{% endfor %}
IMPORTANT: Your goal is to generate new variable that score higher than all past iterations.
<END_OF_HISTORY_PERFORMANCE>
{% endif %}
{# Multi-proposal history #}
{% if failed_proposals %}
<START_OF_CURRENT_ITERATION>
same batch, same feedback: Here are the values you have tried that have not improved the score.(scored <= {{best_score}}):
{% for failed_proposal in failed_proposals %}
{{loop.index}}. {{failed_proposal}}
{% endfor %}
You MUST approach differently from the above methods.
<END_OF_CURRENT_ITERATION>
{% endif %}
{# Feedback #}
{% if variable_grad %}
<START_OF_CONTEXT_FEEDBACK>
Here are the context and feedback for the variable:
{{variable_grad}}
<END_OF_CONTEXT_FEEDBACK>
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

<CRITICAL_INSTRUCTION>
IMPORTANT: Optimize ONLY the content in the TARGET_CONTENT_TO_OPTIMIZE section above.
Do NOT include any references to optimization steps, iterations, or meta-instructions in your response.
Do NOT mention phrases like "when steps exceed", "steps are larger than", "rapid updates", or "step size".
Your output should contain ONLY the improved version of the target content, without any optimization metadata.
</CRITICAL_INSTRUCTION>
<END_OF_USER_MESSAGE>
"""
# NO OPRO history
# TEXT_GRAD_DESC_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
# {{optimizer_system_prompt}}
# <END_OF_SYSTEM_PROMPT>
# <START_OF_USER_MESSAGE>
# You are {{steps}} steps since your last improvement.
# Update the value more rapidly when steps are larger than 3.
# {# Variable and peers info #}
# <START_OF_VARIABLE_AND_PEERS_INFO>
# {{variable_and_peers_info}}
# <END_OF_VARIABLE_AND_PEERS_INFO>
# {# system trainable variables #}
# {% if system_variables %}
# <START_OF_SYSTEM_VARIABLES>
# The target variable is used together with these system variables besides of its peers:
# {% for system_variable in system_variables %}
# {{loop.index}}.
# Name: {{system_variable.name}}
# Type: {{system_variable.param_type}}
# Description: {{system_variable.role_desc}}
# WILL_BE_OPTIMIZED: {{system_variable.requires_opt}}
# Vaule: {{system_variable.prompt_data}}
# {% endfor %}
# Strategically plan the role of each system variable to collaborate with each other for final correct answer.
# <END_OF_SYSTEM_VARIABLES>
# {% endif %}
# {# Feedback #}
# {% if variable_grad %}
# <START_OF_CONTEXT_FEEDBACK>
# Here are the context and feedback for the variable:
# {{variable_grad}}
# <END_OF_CONTEXT_FEEDBACK>
# {% endif %}
# {# Constraints #}
# {% if constraint_text %}
# You must follow the following constraints:
# <CONSTRAINTS>{{constraint_text}}</CONSTRAINTS>
# {% endif %}
# {# In-context examples #}
# {% if in_context_examples %}
# You must base on the following examples when modifying the {{variable_desc}}:
# <EXAMPLES>{{in_context_examples}}</EXAMPLES>
# {% endif %}
# <END_OF_USER_MESSAGE>
# """
# NO promt engineering techniques
# OPTIMIZER_SYSTEM_PROMPT = r"""You are an excellent prompt engineer tasked with instruction and demonstration tuning a compound LLM system.
# Your task is to refine a variable/prompt based on feedback from a batch of input data points.

# The variable is either input or output of a functional component where the component schema will be provided.
# If the same DataID has multiple gradients, it means this component/variable is called multiple times in the compound system(with a cycle) in the same order as it appears in the gradient list.

# You Must edit the current variable with one of the following editing methods.
# You can not rewrite everything all at once:

# You have Four Editing Methods:
# 1. ADD new elements(instruction) to address each specific feedback.
# 2. ADD Examples (e.g., input-reasoning-answer) for tasks that require strong reasoning skills.
# 3. Rephrase existing instruction(for more clarity), Replace existing sample with another, to address the feedback.
# 4. DELETE unnecessary words to improve clarity.

# Your final action/reasoning  = one of FOUR editing method

# You must stick to these instructions:
# 1. **MUST Resolve concerns raised in the feedback** while preserving the positive aspects of the original variable.
# 2. **Observe past performance patterns** to retain good qualities in the variable and past failed ones to try things differently.
# 3. **System Awareness**: When other system variables are given, ensure you understand how this variable works in the whole system.
# 4. **Peer Awareness**: This variable works together with Peer variables, ensure you are aware of their roles and constraints.
# 5. **Batch Awareness**: You are optimizing a batch of input data, ensure the change applys to the whole batch (except while using demonstration.)

# {{output_format_str}}

# {% if instruction_to_optimizer %}
# **Additional User Instructions**: {{instruction_to_optimizer}}
# {% endif %}
# """

OPTIMIZER_SYSTEM_PROMPT = r"""You are an excellent prompt engineer tasked with instruction and demonstration tuning a compound LLM system.
Your task is to refine a variable/prompt based on feedback from a batch of input data points.

The variable is either input or output of a functional component where the component schema will be provided.
If the same DataID has multiple gradients, it means this component/variable is called multiple times in the compound system(with a cycle) in the same order as it appears in the gradient list.

You Must edit the current variable with one of the following editing methods.
You can not rewrite everything all at once:

You have Four Editing Methods:
1. ADD new elements(instruction) to address each specific feedback.
2. ADD Examples (e.g., input-reasoning-answer) for tasks that require strong reasoning skills.
3. Rephrase existing instruction(for more clarity), Replace existing sample with another, to address the feedback.
4. DELETE unnecessary words to improve clarity.

These SIX prompting techniques can be a helpful direction.
1. Set Context and Role: Establish a specific identity or domain expertise for the AI to guide style, knowledge, and constraints.
2. Be Specific, Clear, and Grammarly correct: Clearly define instructions, desired format, and constraints to ensure accurate and relevant outputs with regards to the feedback.
3. Illicit reasoning: "chain-of-thought" (e.g. "think step by step") helps the model reason better.
4. Examples: Construct examples(e.g., input(optional)-reasoning(required)-answer) especially for tasks that require strong reasoning skills.
5. Leverage Constraints and Formatting: Explicitly direct how the answer should be structured (e.g., bullet points, tables, or tone).
6. Self-Consistency / Verification Prompts: Prompt the model to check its own logic for errors, inconsistencies, or missing details.

Your final action/reasoning  = one of FOUR editing method + one of SIX prompting technique.

You must stick to these instructions:
1. **MUST Resolve concerns raised in the feedback** while preserving the positive aspects of the original variable.
2. **Observe past performance patterns** to retain good qualities in the variable and past failed ones to try things differently.
3. **System Awareness**: When other system variables are given, ensure you understand how this variable works in the whole system.
4. **Peer Awareness**: This variable works together with Peer variables, ensure you are aware of their roles and constraints.
5. **Batch Awareness**: You are optimizing a batch of input data, ensure the change applys to the whole batch (except while using demonstration.)

{{output_format_str}}

{% if instruction_to_optimizer %}
**Additional User Instructions**: {{instruction_to_optimizer}}
{% endif %}
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


@dataclass
class TGDData(DataClass):
    reasoning: str = field(metadata={"desc": "Why the variable is proposed this way"})
    method: str = field(
        metadata={
            "desc": "The final method used to propose the variable (prompting + editing)"
        },
    )

    proposed_variable: str = field(
        metadata={"desc": "The proposed variable"},
        default=None,
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


class CustomizedXMLParser(DataComponent):
    """Custom XML parser for TGD optimizer output with reasoning, method, and proposed_variable fields."""

    def __init__(self):
        super().__init__()
        pass

    def get_output_format_str(self) -> str:
        return """Please provide your response in the following XML format:

<response>
<reasoning>Your reasoning for why the variable is proposed this way</reasoning>
<method>The final method used to propose the variable (e.g. prompting + editing)</method>
<proposed_variable>The proposed variable content</proposed_variable>
</response>

Make sure to include all three fields and properly close all XML tags."""

    def call(self, input: str) -> TGDData:
        """Parse the XML response and extract the three fields, returning TGDData directly."""
        try:
            # Clean the input and extract XML content
            input = input.strip()

            # Try to find the response tags
            start_tag = "<response>"
            end_tag = "</response>"

            start_idx = input.find(start_tag)
            end_idx = input.find(end_tag)

            if start_idx == -1 or end_idx == -1:
                # Fallback: try to parse the entire input as XML
                xml_content = input
            else:
                xml_content = input[start_idx : end_idx + len(end_tag)]

            # Parse XML
            root = ET.fromstring(xml_content)

            # Extract fields
            reasoning_elem = root.find("reasoning")
            method_elem = root.find("method")
            proposed_variable_elem = root.find("proposed_variable")

            reasoning = (
                reasoning_elem.text.strip()
                if reasoning_elem is not None and reasoning_elem.text
                else ""
            )
            method = (
                method_elem.text.strip()
                if method_elem is not None and method_elem.text
                else ""
            )
            proposed_variable = (
                proposed_variable_elem.text.strip()
                if proposed_variable_elem is not None and proposed_variable_elem.text
                else ""
            )

            # Create and return TGDData object directly
            return TGDData(
                reasoning=reasoning, method=method, proposed_variable=proposed_variable
            )

        except ET.ParseError as e:
            log.error(f"XML parsing error: {e}")
            return TGDData(
                reasoning="XML parsing failed", method="Error", proposed_variable=input
            )
        except Exception as e:
            log.error(f"Error parsing XML output: {e}")
            return TGDData(
                reasoning="Parsing failed", method="Error", proposed_variable=input
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
    failed_proposals: Dict[str, List[HistoryPrompt]] = {}  # only need the value
    current_tgd_output: Dict[str, Optional[TGDData]] = (
        {}
    )  # id to output, hold all of the data
    one_parameter_at_a_time: bool

    def __init__(
        self,
        params: ParamsT,
        model_client: "ModelClient",
        model_kwargs: Dict[str, object] = {},
        constraints: List[str] = None,
        optimizer_system_prompt: str = OPTIMIZER_SYSTEM_PROMPT,
        in_context_examples: List[str] = None,  # TODO: in-context examples
        max_past_history: int = 3,
        max_failed_proposals: int = 5,  # quite effective
        steps_from_last_improvement: int = 0,
        one_parameter_at_a_time: bool = False,
    ):
        from adalflow.core.generator import Generator
        from adalflow.core import Prompt

        super().__init__()
        self.params = params  # all parameters of prompts (even if non-trainable)

        self.constraints = constraints or []
        self.data_class = TGDData
        self.output_parser = CustomizedXMLParser()
        self.optimizer_system_prompt = Prompt(
            template=optimizer_system_prompt,
            prompt_kwargs={
                "output_format_str": self.output_parser.get_output_format_str(),
            },
        )
        self.variable_and_peers_info = Prompt(
            template=VARIABLE_AND_PEERS_INFO,
        )
        self.do_constrained = len(self.constraints) > 0
        # self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples or []
        self.do_in_context_examples = len(self.in_context_examples) > 0

        self.llm_optimizer = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=TEXT_GRAD_DESC_TEMPLATE,
            output_processors=self.output_parser,
        )

        self.max_past_history = max_past_history
        self.max_failed_proposals = max_failed_proposals
        self.steps_from_last_improvement = steps_from_last_improvement

        self.target_param_index = None
        self.one_parameter_at_a_time = one_parameter_at_a_time
        # initate the past history for each parameter
        for param in self.params:
            self.params_history[param.id] = []
            self.failed_proposals[param.id] = []
            self.current_tgd_output[param.id] = None

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

    def increment_steps_from_last_improvement(self):
        self.steps_from_last_improvement += 1

    def reset_steps_from_last_improvement(self):
        self.steps_from_last_improvement = 0

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
        # check if the value is already in the history, if so, replace it with the new one
        for i, h in enumerate(self.params_history[param_id]):
            if h.value == history.value:
                self.params_history[param_id].pop(i)
                break
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
        """
        Render history for the optimizer prompt.

        Selects top max_past_history prompts by their average score across
        all evaluations (from trainer's multi-minibatch tracking).

        Returns:
            List of YAML strings for the top prompts
        """
        if param_id not in self.params_history:
            return []

        # Get all prompts in history
        all_prompts = self.params_history[param_id]

        if not all_prompts:
            return []

        # If max_past_history is not set or we have fewer prompts, return all
        if not self.max_past_history or len(all_prompts) <= self.max_past_history:
            return [
                history.to_yaml(exclude=["id", "method", "reasoning"])
                for history in all_prompts
            ]

        # Select top max_past_history by score (already sorted in add_history)
        top_prompts = all_prompts[:self.max_past_history]

        return [
            history.to_yaml(exclude=["id", "method", "reasoning"])
            for history in top_prompts
        ]

    def add_failed_proposal(self):
        """Save a copy of the current value of the parameter in the failed proposals."""
        for param in self.params:
            current_tgd_output = self.current_tgd_output.get(param.id, None)
            failed_proposal = HistoryPrompt(
                id=param.id,
                value=param.data,
                eval_score=None,
                method=(current_tgd_output.method if current_tgd_output else None),
                reasoning=(
                    current_tgd_output.reasoning if current_tgd_output else None
                ),
            )
            self.failed_proposals[param.id].append(failed_proposal)
            if len(self.failed_proposals[param.id]) > self.max_failed_proposals:
                for _ in range(
                    len(self.failed_proposals[param.id]) - self.max_failed_proposals
                ):
                    self.failed_proposals[param.id].pop(0)

    def render_failed_proposals(self, param_id: str) -> List[str]:
        if param_id not in self.failed_proposals:
            return []
        return [
            history.to_yaml(exclude=["id", "eval_score", "value"])
            for history in self.failed_proposals[param_id]
        ]

    def _get_user_prompt_kwargs(self, param: Parameter) -> Dict[str, str]:

        system_params = [
            p.get_param_info()
            for p in self.params
            if p.id != param.id and p not in param.peers
        ]
        log.debug(f"system_params: {system_params}")
        peers_params = [p.get_param_info() for p in param.peers]
        variable_and_peer_info = self.variable_and_peers_info.call(
            variable=param.get_param_info(), peers=peers_params
        )

        variable_grad = param.get_gradients_component_schema(skip_correct_sample=False)

        user_prompt_kwargs = {
            "variable_and_peers_info": variable_and_peer_info,
            "variable_grad": variable_grad,
            # constraints
            "constraint_text": self.constraint_text if self.do_constrained else None,
            # in-context examples
            "in_context_examples": (
                "\n".join(self.in_context_examples)
                if self.do_in_context_examples
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
            "best_score": (
                self.params_history[param.id][0].eval_score
                if self.params_history[param.id]
                else "N/A"
            ),
            "system_variables": system_params,
            "steps": self.steps_from_last_improvement,
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

        # reset the failded proposals
        for param in self.params:
            self.failed_proposals[param.id] = []

    def set_target_param(self):
        # iterate through all indexes in cycle
        if self.target_param_index is None:
            self.target_param_index = 0
        else:
            self.target_param_index = (self.target_param_index + 1) % len(self.params)

    # TODO: in the future can propose multiple values at once
    def propose(self):
        r"""Proposing a value while keeping previous value saved on parameter."""
        if self.proposing:
            raise ValueError("Already proposing a value.")

        log.debug("Proposing a new value.")

        # no cache so that new proposal can be made
        no_cache = True
        # print("Proposing a new value.")

        for idx, param in enumerate(self.params):
            if not param.requires_opt:
                log.info(
                    f"Skipping {param.role_desc} as it does not require optimization."
                )
                continue
            if self.one_parameter_at_a_time and idx != self.target_param_index:
                continue

            system_prompt = self.optimizer_system_prompt(
                param_type=str(param.param_type),
                instruction_to_optimizer=param.instruction_to_optimizer,
            )
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
                log.debug(f"Error in the optimizer: {e}")
                raise e
            if not isinstance(response, GeneratorOutput):
                raise TypeError(f"Wrong response type: {type(response)}")

            prompt_str = self.llm_optimizer.get_prompt(**prompt_kwargs)
            log.debug(f"TGD LLM optimizer prompt: {prompt_str}")
            # Handle CustomizedXMLParser output - it returns TGDData directly
            proposed_data: TGDData = (
                response.data
                if response.data is not None and isinstance(response.data, TGDData)
                else TGDData(
                    reasoning="No reasoning",
                    proposed_variable=response.raw_response,
                    method="No method",
                )
            )
            # save current tgd output data
            self.current_tgd_output[param.id] = proposed_data

            log.debug(f"Response from the optimizer: {response}")
            # if not proposed_data.update:
            #     printc(f"No update is required for {param.name}", color="yellow")
            #     param.propose_data(param.data)
            # else:  # TODO: should always trace the initial data
            improved_variable = proposed_data.proposed_variable
            if (
                improved_variable
                and improved_variable != param.data
                and improved_variable != ""
            ):
                param.propose_data(improved_variable)
            else:
                param.propose_data(param.data)
            param.trace_optimizer(api_kwargs=prompt_str, response=response)

        self.proposing = True

    def revert(self):
        """Revert to the previous value when the evaluation is worse."""
        if not self.proposing:
            raise ValueError("Not proposing a value.")
        for idx, param in enumerate(self.params):
            if not param.requires_opt:
                continue
            if self.one_parameter_at_a_time and idx != self.target_param_index:
                continue
            param.revert_data()
            param.trace_optimizer(api_kwargs=None, response=None)
        self.proposing = False

    def step(self):
        """Discard the previous value and keep the proposed value."""
        if not self.proposing:
            raise ValueError("Not proposing a value.")
        for idx, param in enumerate(self.params):
            if not param.requires_opt:
                continue
            if self.one_parameter_at_a_time and idx != self.target_param_index:
                continue

            param.step_data()

        self.proposing = False
    
    def gumbel_top_k(
        self,
        scores,
        k,
        *,
        probs=False,
        seed=None,
        temperature=1.0,      
        noise_scale=1.0,     
        counts=None,          
        ucb_beta=0.0      
    ):
        """
        Gumbel Top-k sampling with balanced exploration-exploitation.

        Args:
            scores: list/1D array. If probs=False, treated as logits; if probs=True, treated as probabilities.
            k: number of indices to sample (k <= len(scores)).
            probs: True if `scores` are probabilities.
            seed: optional RNG seed.

            temperature (float): temperature scaling. T<1 amplifies differences; T>1 increases randomness.
            noise_scale (float): scale of Gumbel noise. 0 disables stochastic exploration.
            counts (list/array or None): evaluation counts n_i per item (for optional UCB bonus).
            ucb_beta (float): >0 to enable a lightweight UCB bonus: β * sqrt(log(N+1)/(n_i+1)).

        Returns:
            List[int]: indices of the top-k (descending by perturbed score).
        """
        x = np.asarray(scores, dtype=np.float64)
        n_items = x.shape[0]
        if k <= 0:
            return []
        k = min(k, n_items)

        if probs:
            x = np.log(np.clip(x, 1e-20, None))

        T = max(float(temperature), 1e-12)
        x = x / T

        explore_ucb = 0.0
        if counts is not None and ucb_beta > 0.0:
            counts = np.asarray(counts, dtype=np.float64)
            if counts.shape[0] != n_items:
                raise ValueError("counts length must match scores length")
            total = np.maximum(counts.sum(), 1.0)
            explore_ucb = ucb_beta * np.sqrt(np.log(total + 1.0) / (counts + 1.0))
            x = x + explore_ucb

        rng = np.random.default_rng(seed)
        u = rng.uniform(low=1e-20, high=1.0 - 1e-20, size=n_items)
        g = -np.log(-np.log(u))
        y = x + float(noise_scale) * g


        print("Pre gumbel scores (after temp & UCB):", x)
        print("Gumbel noise (scaled):", noise_scale * g)
        print("After gumbel scores:", y)

        topk = np.argpartition(y, -k)[-k:]
        topk = topk[np.argsort(y[topk])[::-1]]
        return topk.tolist()

    def generate_top_k_scoring_function(
        self,
        batch_val: List[float],
        batch_val_acc_list: List[List[int]],
        window: Optional[int] = None,
        k: int = 5,
        epsilon_within: float = 0.0,
        beta: float = 1.0
    ) -> List[int]:
        """
        Generate top-K indices using Gumbel-Max sampling.

        This implements Softmax Acquisition via Gumbel-Max:
        - Add Gumbel noise to historical scores
        - Select top-K by perturbed scores

        Based on: https://arxiv.org/pdf/2106.12059

        Args:
            batch_val: List of average validation scores (percentages) for each historical prompt.
                       These are the average accuracies across multiple mini-batch evaluations.
            batch_val_acc_list: List of lists containing individual success/fail records.
                                Not used in this implementation but kept for compatibility.
            window: Optional window size (not used, kept for compatibility)
            k: Number of top prompts to select
            epsilon_within: Epsilon for within-batch exploration (not used)
            beta: Temperature parameter for Gumbel distribution (not used, default 1.0)

        Returns:
            List of indices (with size <= k) in descending order by Gumbel values
        """
        # Convert percentages to probabilities [0, 1]
        scores = [s / 100.0 for s in batch_val]

        # Use Gumbel-Top-K to select indices
        # indices = self.gumbel_top_k(scores=scores, k=k, probs=True)
        indices = self.gumbel_top_k(scores=scores, k=3, probs=False, temperature=0.6, noise_scale=0.4, seed=42)

        return indices

    def top_k_selected_prompts(
        self,
        batch_val: List[float],
        batch_val_acc_list: List[List[int]],
        k: Optional[int] = None
    ):
        """
        Select top-K prompts using Gumbel-Top-K sampling.

        This is the main entry point for Gumbel-based prompt selection.
        It's called during the optimization loop to probabilistically
        select promising historical prompts for refinement.

        Args:
            batch_val: List of average validation scores (percentages) for each
                      historical prompt. The list index corresponds to the prompt
                      iteration number.
            batch_val_acc_list: List of lists containing individual success/fail
                               records for each prompt.
            k: Number of top prompts to select (defaults to self.max_past_history
               if available, otherwise 3)

        Returns:
            Tuple of:
            - selected_prompts: List of selected prompt strings
            - selected_indices: List of selected prompt indices
            - selected_metadata: Optional metadata (None for base implementation)
        """
        if k is None:
            k = getattr(self, 'max_past_history', 3)

        # Get all params that require optimization
        optimizable_params = [p for p in self.params if p.requires_opt]

        if not optimizable_params:
            log.warning("No optimizable parameters found")
            return [], [], None

        # For now, work with the first optimizable parameter
        param = optimizable_params[0]

        # Get the history
        if param.id not in self.params_history:
            log.warning(f"No history found for parameter {param.id}")
            return [str(param.data)], [0], None

        history = self.params_history[param.id]

        if not history:
            # No history yet, return current parameter
            return [str(param.data)], [0], None

        if len(history) == 1:
            # Only one prompt in history, return it
            return [history[0].value], [0], None

        # Ensure batch_val matches history length
        if len(batch_val) != len(history):
            log.warning(
                f"batch_val length ({len(batch_val)}) does not match "
                f"history length ({len(history)}). Using history scores."
            )
            # Use historical scores instead
            batch_val = [h.eval_score for h in history]

        # Perform Gumbel-Top-K selection
        selected_indices = self.generate_top_k_scoring_function(
            batch_val=batch_val,
            batch_val_acc_list=batch_val_acc_list,
            k=min(k, len(history))
        )

        # Extract selected prompts
        selected_prompts = [history[i].value for i in selected_indices]

        log.info(
            f"Selected {len(selected_prompts)} prompts via Gumbel-Top-K "
            f"from {len(history)} candidates"
        )
        log.debug(f"Selected indices: {selected_indices}")

        return selected_prompts, selected_indices, None

    def to_dict(self):
        return {
            "template": TEXT_GRAD_DESC_TEMPLATE,
            "optimizer_system_prompt": OPTIMIZER_SYSTEM_PROMPT,
            "VARIABLE_AND_PEERS_INFO": VARIABLE_AND_PEERS_INFO,
            "params": self.params,
            "constraints": self.constraints,
            "params_history": self.params_history,
            "failed_proposals": self.failed_proposals,
            "max_past_history": self.max_past_history,
            "max_failed_proposals": self.max_failed_proposals,
            "steps_from_last_improvement": self.steps_from_last_improvement,
        }


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
