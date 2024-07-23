"""Implementation of TextGrad: Automatic “Differentiation” via Text"""

from typing import Union
from lightrag.core import Component, Generator, ModelClient, Parameter
from lightrag.core.types import GeneratorOutput
from lightrag.core.functional import compose_model_kwargs
from typing import Dict
from copy import deepcopy


TEXT_LOSS_TEMPLATE = r"""<SYS>
{{sys_prompt_str}}
</SYS>
<USER>
{{user_prompt_str}}
</USER>
You:
"""
# run LLMAsTextLoss with l = LLMAsTextLoss(prompt_kwargs=prompt_kwargs) -> loss
# has a generator, returns a variable after call (dont need forward)
# Example
# sys_prompt_str = "Evaluate the correctness of this sentence"
# user_prompt_str = "A sntence with a typo"
# loss = The sentence you provided does indeed contain a typo. The word "sntence" should be corrected to "sentence."
# it is simplify just a generator and it returns a variable


class LLMAsTextLoss(Component):
    def __init__(
        self,
        prompt_kwargs: Dict[str, Union[str, Parameter]],
        model_client: ModelClient,
        model_kwargs: Dict[str, object],
    ):
        super().__init__()
        prompt_kwargs = deepcopy(prompt_kwargs)
        for key, value in prompt_kwargs.items():
            if isinstance(value, str):
                prompt_kwargs[key] = Parameter(data=value, requires_grad=False)
        self.prompt_kwargs = prompt_kwargs
        # TODO: adapt generator to take both str and Parameter as input
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=TEXT_LOSS_TEMPLATE,
            prompt_kwargs=prompt_kwargs,
        )

    def call(self, prompt_kwargs: Dict[str, Parameter]) -> Parameter:
        combined_prompt_kwargs = compose_model_kwargs(self.prompt_kwargs, prompt_kwargs)
        output: GeneratorOutput = self.generator(prompt_kwargs)
        data = output.data
        response = Parameter(
            data=data,
            requires_grad=False,
            predecessors=list(combined_prompt_kwargs.values()),
            role_desc="response from the language model",
        )
        return response


GRADIENT_TEMPLATE = r"""Here is a conversation:
<CONVERSATION>{{context}}</CONVERSATION>
This conversation is potentially part of a larger system. The output is used as {{response_desc}}
Here is the feedback we got for {{variable_desc}} in the conversation:
    <FEEDBACK>{{feedback}}</FEEDBACK>"""

FEEDBACK_ENGINE_TEMPLATE = r"""<SYS>
{#Overall optimizer description#}
You are part of an optimization system that improves a given text (i.e. the variable).

{#This llm's role in the overal system#}
You are the gradient (feedback) engine.
{#Task specifics#}
- Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags.
- The variables may be solutions to problems, prompts to language models, code, or any other text-based variable.
- Pay attention to the role description of the variable, and the context in which it is used.
- You should assume that the variable will be used in a similar context in the future.
- Only provide strategies, explanations, and methods to change in the variable.
- DO NOT propose a new version of the variable, that will be the job of the optimizer.
- Your only job is to send feedback and criticism (compute 'gradients').
    For instance, feedback can be in the form of 'Since language models have the X failure mode...', 'Adding X can fix this error because...', 'Removing X can improve the objective function because...', 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.
    If a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should not give feedback.

### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.
# - <VARIABLE>: Specifies the span of the variable.
# - <ROLE>: The role description of the variable.
</SYS>
<USER>
You will give feedback to a variable with the following role:
    <ROLE> {{role_desc_str}} </ROLE>.

Here is an evaluation of the variable using a language model:
    <LM_SYSTEM_PROMPT> {{llm_sys_prompt_str}} </LM_SYSTEM_PROMPT>
    <LM_INPUT> {{llm_inputs_str}} </LM_INPUT>
    <LM_OUTPUT> {{llm_pred_str}} </LM_OUTPUT>
    <OBJECTIVE_FUNCTION> {{objective_function_str}} </OBJECTIVE_FUNCTION>

We are interested in giving feedback to the The system prompt for this conversation.
Specifically, give feedback to the following span of text:
<VARIABLE> {{variable_str}} </VARIABLE>

Given the above history, describe how the The system prompt could be improved to improve the <OBJECTIVE_FUNCTION>.
Be very creative, critical, and intelligent.
</USER>
You:
"""
# Example:
role_desc_str = "The system prompt"
llm_sys_prompt_str = "Evaluate the correctness of this sentence"
llm_inputs_str = "A sntence with a typo"
llm_pred_str = """The sentence you provided does indeed contain a typo. The word "sntence" should be corrected to "sentence."""
objective_function_str = "Your goal is to give feedback and criticism to the variable given the above evaluation output. Our only goal is to improve the above metric, and nothing else. "
variable_str = "Evaluate the correctness of this sentence"

# Examoles:
# loss

# NOTE: apply only on variables that require gradient optimization
TEXT_GRAD_OPTIMIZER_TEMPLATE = r"""<SYS>
{#Overall optimizer description#}
You are part of an optimization system that improves text (i.e., variable).
{#This llm's role in the overal system#}
You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable.
You will receive some feedback, and use the feedback to improve the variable.
{#Task specifics#}
- The feedback may be noisy, identify what is important and what is correct.
- Pay attention to the role description of the variable, and the context in which it is used.
- This is very important: You MUST give your response by sending the improved variable between <IMPROVED_VARIABLE> {improved variable} </IMPROVED_VARIABLE> tags.
  The text you send between the tags will directly replace the variable.

### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable.
</SYS>
<USER>
Here is the role of the variable you will improve:
    <ROLE>{{role_desc_str}}</ROLE>.
The variable is the text within the following span:
    <VARIABLE> {{variable_str}} </VARIABLE>
Here is the context and feedback we got for the variable:
    <CONTEXT>
        {#conext is the same as the gradient template#}
        Here is a conversation:
        <CONVERSATION>
            <LM_SYSTEM_PROMPT> {{llm_sys_prompt_str}} </LM_SYSTEM_PROMPT>
            <LM_INPUT> {{llm_inputs_str}} </LM_INPUT>
            <LM_OUTPUT> {{llm_pred_str}} </LM_OUTPUT>
        </CONVERSATION>
        This conversation is potentially part of a larger system. The output is used as response from the language model
        Here is the feedback we got for The input sentence in the conversation:
        <FEEDBACK> {{feedback_str}}</FEEDBACK>
    </CONTEXT>

- Improve the variable (The input sentence) using the feedback provided in <FEEDBACK> tags.
- Send the improved variable in the following format:
    <IMPROVED_VARIABLE>{the improved variable}</IMPROVED_VARIABLE>
- Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else.
</USER>
You:
"""
# Example:
optimizer_role_desc_str = "The input sentence"
variable_str = "A sntence with a typo"
llm_sys_prompt_str = "Evaluate the correctness of this sentence"
llm_inputs_str = "A sntence with a typo"
llm_pred_str = """The sentence you provided does indeed contain a typo. The word "sntence" should be corrected to "sentence."""
feedback_str = 'Since the language model correctly identified a typo in the sentence provided, the feedback for the variable "A sntence with a typo" would be to ensure that the text is free of any spelling errors before presenting it. One way to improve the variable is to run a spell check or proofread the text to catch any typos or spelling mistakes before using it in a context where accuracy is crucial. By ensuring that the text is error-free, the overall quality and credibility of the content will be enhanced, leading to better performance according to the objective function.'

# Note: the separation of the sytem and user prompt fits well that the system prompt is fixed between calls
# x is the parameter?
# 1. use dataclass and structured output to get the output [potentially]
# 2. input is a variable and the system prompt (no gradient?, it is to optimize the system prompt) is another variable -> parameters
