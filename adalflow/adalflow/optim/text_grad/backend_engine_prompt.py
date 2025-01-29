"""Meta-prompts for the backward engine.

Optimized from Textual Auto-diff and enhanced with peer variables.

Reference: TextGrad: Automatic “Differentiation” via Text."""

# NOTE: not receive feedback for good performing case is important for performance
# NOTE: having peers is important to keep the scope of the prompt consistent and not cross-reference with other variables
### System prompt and the template is shared by all GradComponent ###

FEEDBACK_ENGINE_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
You are the feedback engine in an optimization system.

Your task is to provide intelligent and creative feedback for the target variable enclosed in <VARIABLE></VARIABLE> tags,
so that the optimizer can optimize this variable to improve the objective enclosed in <OBJECTIVE_FUNCTION></OBJECTIVE_FUNCTION> tags.

1. Focus on the downstream OBJECTIVE without proposing new versions of the variable.
2. Feedback examples: "Since language models have the X failure mode...", "Adding X can fix this error because...", "Removing X can improve the objective function because...", "Changing X to Y would fix the mistake..."
3. Consider the variable in the context of its peers if provided.

Remember:
Be specific, concise, critical, and direct.
<END_OF_SYSTEM_PROMPT>
<CONVERSATION>
{{conversation_sec}}
</CONVERSATION>
{{objective_instruction_sec}}
"""
##############################################
# Loss Component
##############################################


# Objective instruction for LLM as gradComponent with user custom instruction

# OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
# Our only goal is to improve the above metric, and nothing else.
# {% if instruction_to_backward_engine %}
# Note: {{instruction_to_backward_engine}}
# {% endif %}
# </OBJECTIVE_FUNCTION>"""

OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
Your only goal is to clearly states how it obtained the "<OUTPUTS/SCORE>".
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

<EVAL_FUNC_DESCRIPTION>: {{eval_fn_desc}}
<INPUTS>: {{input_str}}
<OUTPUTS/SCORE>: {{response_value}}
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


# # When the parameter has no gradient, it is the start of the backpropagation chain, used as a loss function
# CONVERSATION_START_INSTRUCTION_BASE = r"""
# {{variable_and_peers_info}}

# Here is an evaluation of the variable using a language model:
# {{conversation_str}}
# """
