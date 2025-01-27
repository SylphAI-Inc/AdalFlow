"""Meta-prompts for the backward engine.

Optimized from Textual Auto-diff and enhanced with peer variables.

Reference: TextGrad: Automatic “Differentiation” via Text."""

# NOTE: not receive feedback for good performing case is important for performance
# NOTE: having peers is important to keep the scope of the prompt consistent and not cross-reference with other variables
### System prompt and the template is shared by all GradComponent ###


FEEDBACK_ENGINE_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
You MUST determining the root cause of a system error.
You start with an evaluation function that measures performance, and you receive the system input.
The system can be a a compound system, potentially consisting of multiple components.
You work on one component.
You will receive feedback from your direct successor component, and your goal is to investigate your component’s inputs and outputs to identify whether any of your input variables are causing the error.

Your target input variable is enclosed in <TARGET_VARIABLE> (representing one of the input variables that may or may not be causing the error).
Alternatively, it may be enclosed in <VARIABLES> tags (in which case you must pass feedback to all variables, indicating which ones cause the errors and which do not).

1. From <CONVERSATION></CONVERSATION> section, you can find how the variable is obtained and used.
2. As there might be multiple precedessors, and multi-components, it is possible that the feedback/error is not directly related to the variable itself.
3. When you reason, really think about the variable's role in the component(infer from the CONVERSATION section) and the VARIABLE section before you provide feedback.
4. Be specific, concise, critical, and direct.
5. Maximum 3 sentences.

[Cycle]: If the same DataID has multiple gradients, it means this component/variable is called multiple times in the compound system(with a cycle) in the same order as it appears in the gradient list.
   Ensure the feedback is aware of all sets of inputs and outputs.

{% if output_format_str %}
{{output_format_str}}
{% endif %}

<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
<CONVERSATION>
{{conversation_sec}}
</CONVERSATION>
<OBJECTIVE_INSTRUCTION>
{{objective_instruction_sec}}
</OBJECTIVE_INSTRUCTION>
<END_OF_USER>
"""
# 6. If you receive error, must find one pred with error!
# 7. Ignore other metadata(noise such as id, data_id) in the data structure, only use the key(input) and key output that matters to infer the component functionality.

##############################################
# Loss Component
##############################################


OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
Your task is to provide the response with specific feedback based on the expected correct response (y_gt/ground_truth) and the score in the "<OUTPUTS/SCORE>".
Especially when the score is low.
Be CONCISE.

Be specific on why it has a low score.
Specify the difference between the expected correct response and the response.
</OBJECTIVE_FUNCTION>"""


OBJECTIVE_INSTRUCTION_CHAIN = r"""This conversation is part of a larger system. The <INPUTS/SCORE> was later used as "{{response_name}}: {{response_desc}}".
<OBJECTIVE_FUNCTION>
Your only goal is to clearly state how it obtained the "Eval output/score": {{response_gradient}}.
Especially when the score is low.
Be CONCISE.
If you have enough context, add more specific feedback on how it failed.
e.g. "The retrieved context is not enough to answer the question so the problem relies on the retrieval part."
</OBJECTIVE_FUNCTION>"""

###  Loss/Score Information  ###


# TODO: add predecessors awareness similar to CONVERSATION_START_INSTRUCTION_CHAIN
# used by GradComponent and loss component
LOSS_CONVERSATION_START_INSTRUCTION_STRING_FN = r"""
TARGET VARIABLE:
<NAME> {{variable.name}} </NAME>
<ROLE> {{variable.role_desc}} </ROLE>
<VARIABLE> {{variable.prompt_data}} </VARIABLE>
{{conversation_str}}
"""

# template to generate the conversation_str in loss component
LOSS_CONVERSATION_TEMPLATE_STRING = r"""
The variable is passed to the eval function and compared with a target/ground truth value to get
its score regarding to a SYSTEM_QUESTION: {{system_question}}.

EVAL_FUNC: {{eval_fn_desc}}

INPUTS to EVAL_FUNC:
{% for key, (value, eval_type) in inputs.items() %}
({{ key }}) (role: {{ value.role_desc }}),
data: {{ value.prompt_data }},
input_to_eval_fn: {{ value.eval_input }},
data_type: {{ eval_type }}
{% endfor %}

OUTPUTS/SCORE: {{response_value}}
{% if metadata %}
Note: {{metadata}}
{% endif %}"""

# template to generate the conversation_str
GRAD_COMPONENT_CONVERSATION_TEMPLATE_STRING = r"""
COMPONENT_DESC: {{component_desc}}

INPUTS:
{% for key, (value, eval_type) in inputs.items() %}
{{loop.index}}.
KEY: {{ key }}.
ROLE: {{ value.role_desc }},
DATA: {{ value.prompt_data }},
{% endfor %}

OUTPUT: {{response_value}}
{% if metadata %}
Note: {{metadata}}
{% endif %}"""


##############################################
# LLM as gradComponent
##############################################
# When the parameter has a gradient, it is the continuation of the backpropagation chain, a layer in the models
CONVERSATION_START_INSTRUCTION_CHAIN = r"""
{{variable_and_peers_info}}

{# system trainable variables #}
{% if predecessors %}
<START_OF_PREDECESSORS>
The target variable is used together with these predecessors variables besides of the peers:
{% for system_variable in predecessors %}
{{loop.index}}.
Name: {{system_variable.name}}
Type: {{system_variable.param_type}}
Description: {{system_variable.role_desc}}
WILL_BE_OPTIMIZED: {{system_variable.requires_opt}}
Value: {{system_variable.prompt_data}}
{% endfor %}
<END_OF_PREDECESSORS>
{% endif %}

Here is the inputs and output with this component(LM):
{{conversation_str}}
"""

# For the generator in the chain,
OBJECTIVE_INSTRUCTION_CHAIN = r"""
This component is part of a larger system. The <LM_OUTPUT> was later used as {{response_desc}}.
<OBJECTIVE_FUNCTION>
Your goal is to give feedback to the variable to guide the LLM_OUTPUT according to feedback: {{response_gradient}}
{% if instruction_to_backward_engine %}
Note: {{instruction_to_backward_engine}}
{% endif %}
</OBJECTIVE_FUNCTION>"""


VARIABLE_AND_PEERS_INFO = r"""
<START_OF_VARIABLE_DESC>
<NAME> {{variable.name}} </NAME>
<TYPE> {{variable.param_type}} </TYPE>
<ROLE> {{variable.role_desc}} </ROLE>
<VARIABLE>{{ variable.prompt_data}}</VARIABLE>
<END_OF_VARIABLE_DESC>
{% if peers %}
<VARIBLE_PEERS>
{% for peer in peers %}
{{loop.index}}.
PEER_NAME: {{peer.name}},
PEER_TYPE: {{peer.param_type}},
PEER_ROLE: {{peer.role_desc}}
WILL_BE_OPTIMIZED: {{peer.requires_opt}}
{% if peer.prompt_data %}
PEER_VARIABLE: {{peer.prompt_data}}
{% else %}
PEER_VARIABLE: EMPTY
{% endif %}
{% endfor %}
</VARIBLE_PEERS>
{% endif %}
"""

# The variable is used together with the these peer variables to instruct the language model on the task.
# - Do not overlap with the scope of the peer.


# a list of variables
ALL_PRED_INFO = r"""
<VARIABLES>
{% if variables %}
Length of the list: {{variables|length}}
{% for variable in variables %}
{{loop.index}}.
NAME: {{variable.name}},
TYPE: {{variable.param_type}},
ROLE: {{variable.role_desc}}
WILL_BE_OPTIMIZED: {{variable.requires_opt}}
VARIABLE: {{ variable.prompt_data}}
{% endfor %}
{% endif %}
</VARIABLES>
"""


###  Backward engine: user prompt
# First part to provide context of LLM as gradComponent
# The target variable is used as either input or a task instruction to a language model (LM):
# replace the "The target variable is used as either input or a task instruction to a language model (LM):" with the {{variable_desc}}
# NAME: {{variable_name}}
# Description: {{variable_desc}}
LLM_CONVERSATION_TEMPLATE = r"""
LM_INPUT: {{input_value}}
LM_OUTPUT: {{llm_output}}
{% if gt %}
GROUND_TRUTH: {{gt}}
{% endif %}
"""

# OUTPUT_INSTRUCTION = r"""
# You will create a feedback for each of the variable in the list above.
# If a variable will not be optimied, you just output empty string for that variable..
# NOTE: you MUST output a list of strings with the same length as the list above as ["...", "...", "..."]
# """
OUTPUT_INSTRUCTION = r"""
You will create a feedback for each of the variables in the list.
If a variable will not be optimized, you just output empty string.
Give enough details on the feedback.
Your output will be a list of strings with the SAME LENGTH as the <VARIABLES> list
as format of ["...", "...", "..."]
"""
