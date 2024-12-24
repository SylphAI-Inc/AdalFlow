"""Meta-prompts for the backward engine.

Optimized from Textual Auto-diff and enhanced with peer variables.

Reference: TextGrad: Automatic “Differentiation” via Text."""

# NOTE: not receive feedback for good performing case is important for performance
# NOTE: having peers is important to keep the scope of the prompt consistent and not cross-reference with other variables
### System prompt and the template is shared by all GradComponent ###

FEEDBACK_ENGINE_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
You are the feedback engine in an optimization system consits of multiple components.

Your task is to provide intelligent and creative feedback in each component for the target variable enclosed in <VARIABLE></VARIABLE> tags,
so that the optimizer can optimize this variable to improve the objective enclosed in <OBJECTIVE_FUNCTION></OBJECTIVE_FUNCTION> tags.

1. Focus on the downstream OBJECTIVE without proposing new versions of the variable.
2. From <CONVERSATION></CONVERSATION> section, you can find how the variable is obtained and used.
3. The variable might have other peers that are used together to instruct the language model. But only focus on the target variable.
4. As there might be peers, and multi-components, it is possible that the feedback/error is not directly related to the variable itself.
In such cases, you can just say "There is no noticeable error".
5. When you reason, really think about the variable's role in the component(infer from the CONVERSATION section) and the VARIABLE section before you provide feedback.
6. Be specific, concise, critical, and direct.
<END_OF_SYSTEM_PROMPT>
<CONVERSATION>
{{conversation_sec}}
</CONVERSATION>
{{objective_instruction_sec}}
{% if output_format_str %}
{{output_format_str}}
{% endif %}
"""
##############################################
# Loss Component
##############################################
# 2. Feedback examples: "Since language models have the X failure mode...", "Adding X can fix this error because...", "Removing X can improve the objective function because...", "Changing X to Y would fix the mistake..."

# Objective instruction for LLM as gradComponent with user custom instruction

# OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
# Our only goal is to improve the above metric, and nothing else.
# {% if instruction_to_backward_engine %}
# Note: {{instruction_to_backward_engine}}
# {% endif %}
# </OBJECTIVE_FUNCTION>"""
# Your only goal is to clearly states how it obtained the "<OUTPUTS/SCORE>".

OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
Your task is to provide the response with specific feedback based on the ground truth and the score in the "<OUTPUTS/SCORE>".
Especially when the score is low.
Be CONCISE.
Be specific on why it has a low score.
e.g. "The retrieved context is not enough to answer the question so the problem relies on the retrieval part."
</OBJECTIVE_FUNCTION>"""


### Variable to get feedback on, often it is pred in the loss component
LOSS_CONVERSATION_START_INSTRUCTION_STRING_FN = r"""
TARGET VARIABLE:
<NAME> {{variable_name}} </NAME>
<ROLE> {{variable_desc}} </ROLE>
<VARIABLE> {{variable_value}} </VARIABLE>
{{conversation_str}}
"""

###  Loss/Score Information  ###
LOSS_CONVERSATION_TEMPLATE_STRING = r"""
The variable is passed to the eval function and compared with a target/ground truth value.

EVAL_FUNC: {{eval_fn_desc}}

INPUTS:
{% for key, (value, eval_type) in inputs.items() %}
({{ key }}) (role: {{ value.role_desc }}),
full response: {{ value.data }},
input_to_eval_fn: {{ value.eval_input }},
data_type: {{ eval_type }}
{% endfor %}

OUTPUTS/SCORE: {{response_value}}
{% if metadata %}
Note: {{metadata}}
{% endif %}"""


##############################################
# LLM as gradComponent
##############################################
# When the parameter has a gradient, it is the continuation of the backpropagation chain, a layer in the models
CONVERSATION_START_INSTRUCTION_CHAIN = r"""
{{variable_and_peers_info}}

Here is a conversation with the language model (LM):
{{conversation_str}}
"""

# For the generator in the chain,
OBJECTIVE_INSTRUCTION_CHAIN = r"""
This conversation is part of a larger system. The <LM_OUTPUT> was later used as {{response_desc}}.
<OBJECTIVE_FUNCTION>
Your goal is to give feedback to the variable to guide the LLM_OUTPUT according to feedback: {{response_gradient}}
{% if instruction_to_backward_engine %}
Note: {{instruction_to_backward_engine}}
{% endif %}
</OBJECTIVE_FUNCTION>"""

###  Backward engine: user prompt
# First part to provide context of LLM as gradComponent
# The target variable is used as either input or a task instruction to a language model (LM):
# replace the "The target variable is used as either input or a task instruction to a language model (LM):" with the {{variable_desc}}
# NAME: {{variable_name}}
# Description: {{variable_desc}}
LLM_CONVERSATION_TEMPLATE = r"""
LM_INPUT: {{input_value}}
LM_OUTPUT: {{llm_output}}"""


VARIABLE_AND_PEERS_INFO = r"""
<START_OF_VARIABLE_DESC>
{{variable.name}}
<TYPE> {{variable.param_type}} </TYPE>
<ROLE> {{variable.role_desc}} </ROLE>
<VARIABLE> {{variable.data}} </VARIABLE>
<END_OF_VARIABLE_DESC>
{% if peers %}
<VARIBLE_PEERS>
The variable is used together with the these peer variables to instruct the language model:
{% for peer in peers %}
{{loop.index}}.
PEER_NAME: {{peer.name}},
PEER_TYPE: {{peer.param_type}},
PEER_ROLE: {{peer.role_desc}}
WILL_BE_OPTIMIZED: {{peer.requires_opt}}
{% if peer.data %}
PEER_VARIABLE: {{peer.data}}
{% else %}
PEER_VARIABLE: EMPTY
{% endif %}
{% endfor %}
</VARIBLE_PEERS>
{% endif %}
"""

# a list of variables
ALL_PRED_INFO = r"""
<VARIABLES>
{% for variable in variables %}
{{loop.index}}.
NAME: {{variable.name}},
TYPE: {{variable.param_type}},
ROLE: {{variable.role_desc}}
WILL_BE_OPTIMIZED: {{variable.requires_opt}}
VARIABLE: {{variable.data}}
{% endfor %}"""

OUTPUT_INSTRUCTION = r"""
You will create a feedback for each of the variable in the list above.
If a variable will not be optimied, you just output empty string.
Your output will be a list of strings with the same length as the list above.
"""


# # When the parameter has no gradient, it is the start of the backpropagation chain, used as a loss function
# CONVERSATION_START_INSTRUCTION_BASE = r"""
# {{variable_and_peers_info}}

# Here is an evaluation of the variable using a language model:
# {{conversation_str}}
# """

##############################################
# Backward multiple peers at the same time
##############################################
