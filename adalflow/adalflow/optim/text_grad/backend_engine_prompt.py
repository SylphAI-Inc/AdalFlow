"""Meta-prompts for the backward engine.

Optimized from Textual Auto-diff and enhanced with peer variables.

Reference: TextGrad: Automatic “Differentiation” via Text."""

# NOTE: not receive feedback for good performing case is important for performance
# NOTE: having peers is important to keep the scope of the prompt consistent and not cross-reference with other variables
### System prompt and the template is shared by all GradComponent ###

# FEEDBACK_ENGINE_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
# You are the feedback engine in an optimization system consisting of multiple components.

# Your task is to provide intelligent and creative feedback in each component for the target variable enclosed in <TARGET_VARIABLE> or <VARIABLES> tags
# so that the optimizer can optimize this variable to improve the objective enclosed in <OBJECTIVE_FUNCTION> tags.

# Instructions:
# 1. Understand the role of each variable in the component system BEFORE you give feedback.
# 2. You MUST attribute the feedback to the correct variable only.
# 3. Focus on the downstream objective without proposing new versions of the variable.
# 4. From the <CONVERSATION> section, see how the variable is obtained and used.
# 5. The variable might have peers also used to instruct the language model, but your feedback should only focus on the target variable.
# 6. If the error is not directly related to the variable itself, you can say: \"There is no noticeable error.\"
# 7. Be specific, concise, critical, and direct.
# 8. If the same DataID appears multiple times, it means the component/variable is called repeatedly in the same order as it appears in the gradient list.


# {% if output_format_str %}
# {{output_format_str}}
# {% endif %}

# <END_OF_SYSTEM_PROMPT>
# <START_OF_USER>
# <CONVERSATION>
# {{conversation_sec}}
# </CONVERSATION>
# <OBJECTIVE_INSTRUCTION>
# {{objective_instruction_sec}}
# </OBJECTIVE_INSTRUCTION>
# <END_OF_USER>
# """

FEEDBACK_ENGINE_PEERS_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
You are the feedback engine in an optimization system consisting of multiple components.

A component can have multiple inputs, and you handle one that is enclosed in <TARGET_VARIABLE> or <VARIABLES> tags.
You will provide intelligent and creative feedback so that the optimizer can optimize this variable to improve the objective enclosed in <OBJECTIVE_FUNCTION> tags.

About <VARIABLES> or <PEERS>:
* If a variable is of type "output", it is the output of another predecessor component. In this case, you MUST attribute the error to the RIGHT variable.
* If a variable plays no role to the error, simply state "This variable did not cause the error. No need to change the essense of this variable."

1. From <CONVERSATION></CONVERSATION> section, you can find how the variable is obtained and used.
2. The variable might have other peers that are used together to instruct the language model. But only focus on the target variable.
3. As there might be peers, and multi-components, it is possible that the feedback/error is not directly related to the variable itself.
4. When you reason, really think about the variable's role in the component(infer from the CONVERSATION section) and the VARIABLE section before you provide feedback.
5. Be specific, concise, critical, and direct.


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
# 1. Focus on the downstream OBJECTIVE without proposing new versions of the variable.

# <TASK_PIPELINE>
# Here is a summary on the task pipeline you are optimizing:
# retriever: retrieves relevant documents for the question. (Not trainable, you have no control)
# LLM: Answer questions by reading the context  and reason the best answer.
# </TASK_PIPELINE>
# You are the feedback engine in an optimization system consisting of multiple components.
# You are the feedback engine to provide feedback for a target variable in a compound LLM system.

# The evaluation and feedback is backpropogated all the way to you, and you will assess the current component's inputs, output along with its feedback.
# A component can have multiple inputs, and you handle one that is enclosed in <TARGET_VARIABLE> or <VARIABLES> tags.
# You will provide intelligent and creative feedback so that the optimizer can optimize this variable to improve the objective enclosed in <OBJECTIVE_FUNCTION> tags.

FEEDBACK_ENGINE_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
You are a detective excel at determining the root cause of a system error.
You start with an evaluation function that measures performance, and you receive the system input.
The system can be a a compound system, potentially consisting of multiple components.
You will receive feedback from your direct successor, and your goal is to investigate your component’s inputs and outputs to identify whether any of your input variables are causing the error.

Your target input variable is enclosed in <TARGET_VARIABLE> (representing one of the input variables that may or may not be causing the error).
Alternatively, it may be enclosed in <VARIABLES> tags (in which case you must pass feedback to all variables, indicating which ones cause the errors and which do not).

1. From <CONVERSATION></CONVERSATION> section, you can find how the variable is obtained and used.
2. As there might be multiple precedessors, and multi-components, it is possible that the feedback/error is not directly related to the variable itself.
3. When you reason, really think about the variable's role in the component(infer from the CONVERSATION section) and the VARIABLE section before you provide feedback.
4. [Cycle]: If the same DataID has multiple gradients, it means this component/variable is called multiple times in the compound system(with a cycle) in the same order as it appears in the gradient list.
   Ensure the feedback is aware of all sets of inputs and outputs.
5. Be specific, concise, critical, and direct.
6. Maximum 3 sentences.

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
# In such cases, you can just say "There is no noticeable error".

# 2. Feedback examples: "Since language models have the X failure mode...", "Adding X can fix this error because...", "Removing X can improve the objective function because...", "Changing X to Y would fix the mistake..."

# Objective instruction for LLM as gradComponent with user custom instruction

# OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
# Our only goal is to improve the above metric, and nothing else.
# {% if instruction_to_backward_engine %}
# Note: {{instruction_to_backward_engine}}
# {% endif %}
# </OBJECTIVE_FUNCTION>"""
# Your only goal is to clearly states how it obtained the "<OUTPUTS/SCORE>".


# OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
# Your only goal is to clearly states how it obtained the "<OUTPUTS/SCORE>",
# so that you can inform other components on the specific errors.
# e.g. "The <gt> and <pred> are not an exact match, it differs by <difference>."
# Especially when the score is low.
# Be CONCISE. Be SPECIFIC.
# </OBJECTIVE_FUNCTION>"""

# OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
# Your task: Provide specific feedback based on the score in the \"<OUTPUTS/SCORE>\" value.
# - Especially note when the score is low (e.g. 0.0).
# - Be concise.
# - Be specific about why the score is low. For example:
#   The retrieved context is insufficient to answer the question accurately.
# </OBJECTIVE_FUNCTION>"""

OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
Your task is to provide the response with specific feedback based on the expected correct response (y_gt/ground_truth) and the score in the "<OUTPUTS/SCORE>".
Especially when the score is low.
Be CONCISE.

Be specific on why it has a low score.
Specify the difference between the expected correct response and the response.
</OBJECTIVE_FUNCTION>"""

# Be specific on why it has a low score.

### NOTE: Last node's feedback
# OBJECTIVE_INSTRUCTION_CHAIN = r"""This conversation is part of a larger system. The <INPUTS/SCORE> was later used as "{{response_name}}: {{response_desc}}".
# <OBJECTIVE_FUNCTION>
# Your only goal is to clearly provide feedback on obtaining "Eval output/score": {{response_gradient}}.
# Be CONCISE and specific on how it can be improved.
# </OBJECTIVE_FUNCTION>"""

OBJECTIVE_INSTRUCTION_CHAIN = r"""This conversation is part of a larger system. The <INPUTS/SCORE> was later used as "{{response_name}}: {{response_desc}}".
<OBJECTIVE_FUNCTION>
Your only goal is to clearly states how it obtained the "Eval output/score": {{response_gradient}}.
Especially when the score is low.
Be CONCISE.
If you have enough context, add a more specific feedback on how it failed.
e.g. "The retrieved context is not enough to answer the question so the problem relies on the retrieval part."
</OBJECTIVE_FUNCTION>"""

###  Loss/Score Information  ###
# INPUTS: parameter.get_param_info():
# the input_output of a GradientContext

# response_value -> response.get_prompt_data()
# LOSS_CONVERSATION_TEMPLATE_STRING = r"""
# The target variable is passed to the EVAL_FUNC and compared with the correct value.

# EVAL_FUNC: {{eval_fn_desc}}

# INPUTS:
# {% for key, (value, eval_type) in inputs.items() %}
# ({{ key }}) (role: {{ value.role_desc }}),
# data: {{ value.prompt_data }},
# input_to_eval_fn: {{ value.eval_input }},
# data_type: {{ eval_type }}
# {% endfor %}

# OUTPUTS/SCORE: {{response_value}}
# {% if metadata %}
# Note: {{metadata}}
# {% endif %}"""

# LOSS_CONVERSATION_TEMPLATE_STRING = r"""
# The variable is passed to the eval function and compared with a expected value(y_gt or ground_truth).

# EVAL_FUNC: {{eval_fn_desc}}

# INPUTS:
# {% for key, (value, eval_type) in inputs.items() %}
# ({{ key }}) (role: {{ value.role_desc }}),
# data: {{ value.prompt_data }},
# input_to_eval_fn: {{ value.eval_input }},
# data_type: {{ eval_type }}
# {% endfor %}

# OUTPUTS/SCORE: {{response_value}}
# {% if metadata %}
# Note: {{metadata}}
# {% endif %}"""


### Variable to get feedback on, often it is pred in the loss component
# pass parameter.get_param_info() to get the variable info
LOSS_CONVERSATION_START_INSTRUCTION_STRING_FN = r"""
TARGET VARIABLE:
<NAME> {{variable.name}} </NAME>
<ROLE> {{variable.role_desc}} </ROLE>
<VARIABLE> {{variable.prompt_data}} </VARIABLE>
{{conversation_str}}
"""

###  Loss/Score Information  ###
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
<START_OF_PRECESSORS>
The target variable is used together with these predecessors variables besides of the peers:
{% for system_variable in predecessors %}
{{loop.index}}.
Name: {{system_variable.name}}
Type: {{system_variable.param_type}}
Description: {{system_variable.role_desc}}
WILL_BE_OPTIMIZED: {{system_variable.requires_opt}}
Vaule: {{system_variable.prompt_data}}
{% endfor %}
<END_OF_PRECESSORS>
{% endif %}

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


SUMMARY_TASK = """
Here is a summary on the task pipeline you are optimizing:
query_generator: "generates a sub-query based on the initial query"
retriever: "retrieves relevant documents based on the sub-query"
llm: "Answer a question with available context with exact answer extracted from the context"

The query_generator is called twice in the pipeline.
And the retrieved documents are deduplicated and combined to form the final context.
The final context is then passed to the llm to generate the answer where we want to use the exact phrase from the context.
"""


# VARIABLE_AND_PEERS_INFO = r"""
# <START_OF_VARIABLE_DESC>
# {{variable.name}}
# <TYPE> {{variable.param_type}} </TYPE>
# <ROLE> {{variable.role_desc}} </ROLE>
# <VARIABLE>{{ variable.prompt_data}}</VARIABLE>
# <END_OF_VARIABLE_DESC>
# {% if peers %}
# <VARIBLE_PEERS>
# The variable is used together with the these peer variables to instruct the language model:
# {% for peer in peers %}
# {{loop.index}}.
# PEER_NAME: {{peer.name}},
# PEER_TYPE: {{peer.param_type}},
# PEER_ROLE: {{peer.role_desc}}
# WILL_BE_OPTIMIZED: {{peer.requires_opt}}
# {% if peer.prompt_data %}
# PEER_VARIABLE: {{peer.prompt_data}}
# {% else %}
# PEER_VARIABLE: EMPTY
# {% endif %}
# {% endfor %}
# </VARIBLE_PEERS>
# {% endif %}
# """

VARIABLE_AND_PEERS_INFO = r"""
<START_OF_VARIABLE_DESC>
{{variable.name}}
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
If a variable will not be optimied, you just output empty string.
Give enough details on the feedback.
Your output will be a list of strings with the SAME LENGTH as the <VARIABLES> list
as format of ["...", "...", "..."]
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
