"""Meta-prompts for the backward engine.

Optimized from Textual Auto-diff and enhanced with peer variables.

Reference: TextGrad: Automatic “Differentiation” via Text."""

# NOTE: not receive feedback for good performing case is important for performance
# NOTE: having peers is important to keep the scope of the prompt consistent and not cross-reference with other variables
### System prompt and the template is shared by all GradComponent ###

FEEDBACK_ENGINE_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
You are the feedback engine in an optimization system.

Your role: Provide intelligent and creative feedback for the variable enclosed in <VARIABLE></VARIABLE> tags, based on the objective specified in <OBJECTIVE_FUNCTION></OBJECTIVE_FUNCTION> tags.
1. Focus on the downstream OBJECTIVE without proposing new versions of the variable.
2. Feedback examples: "Since language models have the X failure mode...", "Adding X can fix this error because...", "Removing X can improve the objective function because...", "Changing X to Y would fix the mistake..."
3. Consider the variable in the context of its peers if provided.
Remember:
Be concise, critical, and direct.
<END_OF_SYSTEM_PROMPT>
<CONVERSATION>
{{conversation_sec}}
</CONVERSATION>
{{objective_instruction_sec}}
"""

###  Backward engine: user prompt
# First part to provide context of LLM as gradComponent
LLM_CONVERSATION_TEMPLATE = r"""
NAME: {{variable_name}}
The target variable is used as either input or a task instruction to a language model (LM):

LM_INPUT: {{input_value}}
LM_OUTPUT: {{llm_output}}"""


# only passing variable (dict) and peers as parameters
# shared between the
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
Our only goal is to improve the above metric, and nothing else.
{% if instruction_to_backward_engine %}
Note: {{instruction_to_backward_engine}}
{% endif %}
</OBJECTIVE_FUNCTION>"""


OBJECTIVE_INSTRUCTION_CHAIN = r"""
This conversation is part of a larger system. The <LM_OUTPUT> was later used as {{response_desc}}.
<OBJECTIVE_FUNCTION>
Your goal is to give feedback to the variable with the LLM_OUTPUT: {{response_gradient}}
{% if instruction_to_backward_engine %}
Note: {{instruction_to_backward_engine}}
{% endif %}
</OBJECTIVE_FUNCTION>"""
