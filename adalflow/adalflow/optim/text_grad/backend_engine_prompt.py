"""Meta-prompts for the backward engine.
Adapted from TextGrad: Automatic “Differentiation” via Text."""

GLOSSARY_TEXT_BACKWARD = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.
# - <VARIABLE>: Specifies the span of the variable.
# - <ROLE>: The role description of the variable."""


# NOTE: not receive feedback is important for performance
# NOTE: having peers is important to keep the scope of the prompt consistent and not cross-reference with other variables
### System prompt and the template is shared by all GradComponent ###
FEEDBACK_ENGINE_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
You are the feedback engine in a larger optimization system.

Your only responsibility is to give to a variable enclosed by <VARIABLE></VARIBLE> tags intelligent and creative feedback
with regarding to an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags.

The variable may be solution to problems, prompt/instruction to langage model, code, or any other text-based variable.
{#Task specifics#}
Remember:
- DO NOT propose a new version of the variable, that will be the job of the optimizer.
  For instance, feedback can be in the form of 'Since language models have the X failure mode...', 'Adding X can fix this error because...', 'Removing X can improve the objective function because...', 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.
- If a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should respond with "It works well in this case, no critical feedback.
- BE CONCISE (DONOT repeat the variable value), CRITICAL, and CREATIVE.

<END_OF_SYSTEM_PROMPT>

<START_OF_USER_PROMPT>
{{ "\"\"\"" }}{{conversation_sec}}{{ "\"\"\"" }}

{{objective_instruction_sec}}

<END_OF_USER_PROMPT>"""

###  Backward engine: user prompt
# First part to provide context of LLM as gradComponent
LLM_CONVERSATION_TEMPLATE = r"""
Target variable:
Name: {{variable_name}}
Role Description: {{variable_desc}}
Type: {{param_type}}

The target variable is used as either input or a task instruction to a language model (LM):
Input to the LM: {{input_value}}
LLM output: {{llm_output}}"""


# only passing variable (dict) and peers as parameters
# shared between the
VARIABLE_AND_PEERS_INFO = r"""
<START_OF_VARIABLE_DESC>
Variable information:
Name: {{variable.name}}
Type: {{variable.param_type}}
Role Description: {{variable.role_desc}}.
Variable value: <VARIABLE> {{variable.data}} </VARIABLE>

<END_OF_VARIABLE_DESC>
{% if peers %}
<VARIBLE_PEERS>
The variable is used together with the following peer variables:
{% for peer in peers %}
{{loop.index}}.
Name: {{peer.name}},
Type: {{peer.param_type}},
Role Description: {{peer.role_desc}}
{% if peer.data %}
Value: {{peer.data}}
{% endif %}
{% endfor %}
</VARIBLE_PEERS>
{% endif %}
"""

# When the parameter has no gradient, it is the start of the backpropagation chain, used as a loss function
CONVERSATION_START_INSTRUCTION_BASE = r"""
{{variable_and_peers_info}}

Here is an evaluation of the variable using a language model:
{{conversation_str}}
"""

# When the parameter has a gradient, it is the continuation of the backpropagation chain, a layer in the models
CONVERSATION_START_INSTRUCTION_CHAIN = r"""
{{variable_and_peers_info}}

Here is a conversation with a language model (LM):
{{conversation_str}}
"""

# Objective instruction for LLM as gradComponent with user custom instruction
OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
Your goal is to give feedback and criticism to the variable given the above evaluation output.
Our only goal is to improve the above metric, and nothing else.
{% if instruction_to_backward_engine %}
Note: {{instruction_to_backward_engine}}
{% endif %}
</OBJECTIVE_FUNCTION>"""


OBJECTIVE_INSTRUCTION_CHAIN = r"""
This conversation is part of a larger system. The <LM_OUTPUT> was later used as {{response_desc}}.
<OBJECTIVE_FUNCTION>
Your goal is to give feedback to the variable to address the following feedback on the LLM output: {{response_gradient}}
{% if instruction_to_backward_engine %}
Note: {{instruction_to_backward_engine}}
{% endif %}
</OBJECTIVE_FUNCTION>"""


# Third part pf the user prompt


# EVALUATE_VARIABLE_INSTRUCTION = r"""We are interested in giving feedback to the {{variable_desc}}
# for this conversation. Specifically, give feedback to the following span of text:
# <VARIABLE> {{variable_short}} </VARIABLE>
# Given the above history, describe how the {{variable_desc}}
# could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.
# """


# TODO: Not fully sure about the function of this template
GRADIENT_TEMPLATE = r"""Here is a conversation:
<CONVERSATION>{{context}}</CONVERSATION>
This conversation is potentially part of a larger system. The output is used as {{response_desc}}
Here is the feedback we got for {{variable_desc}} in the conversation:
    <FEEDBACK>{{feedback}}</FEEDBACK>"""
