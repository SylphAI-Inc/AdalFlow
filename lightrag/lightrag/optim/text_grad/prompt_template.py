"""Implementation of TextGrad: Automatic “Differentiation” via Text"""

GLOSSARY_TEXT_BACKWARD = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.
# - <VARIABLE>: Specifies the span of the variable.
# - <ROLE>: The role description of the variable."""

TEXT_LOSS_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{eval_system_prompt}}
<END_OF_SYSTEM_PROMPT>
<USER>
{{eval_user_prompt}}
</USER>
"""

# CONVERSATION_TEMPLATE = (
#     "<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
#     "<LM_INPUT> {prompt} </LM_INPUT>\n\n"
#     "<LM_OUTPUT> {response_value} </LM_OUTPUT>\n\n"
# )

CONVERSATION_TEMPLATE = r"""<LM_SYSTEM_PROMPT> {{eval_system_prompt}} </LM_SYSTEM_PROMPT>
<LM_INPUT> {{eval_user_prompt}} </LM_INPUT>
<LM_OUTPUT> {{response_value}} </LM_OUTPUT>"""


GRADIENT_TEMPLATE = r"""Here is a conversation:
<CONVERSATION>{{context}}</CONVERSATION>
This conversation is potentially part of a larger system. The output is used as {{response_desc}}
Here is the feedback we got for {{variable_desc}} in the conversation:
    <FEEDBACK>{{feedback}}</FEEDBACK>"""


"""
"You are part of an optimization system that improves a given text (i.e. the variable).

You are the gradient (feedback) engine. Your only responsibility is to give intelligent and creative feedback and constructive
criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags.
The variables may be solutions to problems, prompts to language models, code, or any other text-based variable.
Pay attention to the role description of the variable,
and the context in which it is used. You should assume that the variable will be used in a similar context in the future.
Only provide strategies, explanations, and methods to change in the variable. DO NOT propose a new version of the variable,
that will be the job of the optimizer. Your only job is to send feedback and criticism (compute 'gradients').
 For instance, feedback can be in the form of 'Since language models have the X failure mode...',
   'Adding X can fix this error because...', 'Removing X can improve the objective function because...',
   'Changing X to Y would fix the mistake ...', that gets at the downstream objective.\nIf a variable is already working well
   (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should not give feedback.\n\n
   ### Glossary of tags that will be sent to you:\n# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.\n# -
   # <LM_INPUT>: The input to the language model.\n# - <LM_OUTPUT>: The output of the language model.\n# - <OBJECTIVE_FUNCTION>:
   # The objective of the optimization task.\n# - <VARIABLE>: Specifies the span of the variable.\n# - <ROLE>:
   # The role description of the variable."},

{'role': 'user', 'content': '

# CONVERSATION_START_INSTRUCTION_CHAIN
# You will give feedback to a variable with
#    # the following role: <ROLE> The input sentence </ROLE>. Here is an evaluation of the variable using a language model:\n\n
#    # <LM_SYSTEM_PROMPT> Evaluate the correctness of this sentence </LM_SYSTEM_PROMPT>\n\n<LM_INPUT> A sntence with a typo
#    # </LM_INPUT>\n\n<LM_OUTPUT> The sentence you provided does indeed contain a typo.
#    # The word "sntence" should be corrected to "sentence." </LM_OUTPUT>\n\n

# OBJECTIVE_INSTRUCTION_BASE
   # <OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output.
   # Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\n

# EVALUATE_VARIABLE_INSTRUCTION
   # We are interested in giving feedback to the The input sentence for this conversation.
   # Specifically, give feedback to the following span of text:\n\n<VARIABLE> A sntence with a typo </VARIABLE>\n\n
   # Given the above history, describe how the The input sentence could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.\n\n'}]
generate messages [{'role': 'system', 'content': "You are part of an optimization system that improves a given text (i.e. the variable).
You are the gradient (feedback) engine.
Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables,
given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags.
The variables may be solutions to problems, prompts to language models, code, or any other text-based variable.
Pay attention to the role description of the variable, and the context in which it is used.
You should assume that the variable will be used in a similar context in the future.
 Only provide strategies, explanations, and methods to change in the variable.
 DO NOT propose a new version of the variable, that will be the job of the optimizer.
 Your only job is to send feedback and criticism (compute 'gradients').
 For instance, feedback can be in the form of 'Since language models have the X failure mode...',
 'Adding X can fix this error because...', 'Removing X can improve the objective function because...',
 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.\n
 If a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response
 is accurate), you should not give feedback.\n\n### Glossary of tags that will be sent to you:\n# - <LM_SYSTEM_PROMPT>:
   The system prompt for the language model.\n# - <LM_INPUT>: The input to the language model.\n# - <LM_OUTPUT>: Th
   e output of the language model.\n# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.\n# - <VARIAB
   LE>: Specifies the span of the variable.\n# - <ROLE>: The role description of the variable."
"""
# 2024-07-24 20:09:39 - openai - INFO - [openai.py:63:generate] - generate messages: [{'role': 'system', 'content': "You are part of an optimization system that improves a given text (i.e. the variable). You are the gradient (feedback) engine. Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags. The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. Pay attention to the role description of the variable, and the context in which it is used. You should assume that the variable will be used in a similar context in the future. Only provide strategies, explanations, and methods to change in the variable. DO NOT propose a new version of the variable, that will be the job of the optimizer. Your only job is to send feedback and criticism (compute 'gradients'). For instance, feedback can be in the form of 'Since language models have the X failure mode...', 'Adding X can fix this error because...', 'Removing X can improve the objective function because...', 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.\nIf a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should not give feedback.\n\n### Glossary of tags that will be sent to you:\n# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.\n# - <LM_INPUT>: The input to the language model.\n# - <LM_OUTPUT>: The output of the language model.\n# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.\n# - <VARIABLE>: Specifies the span of the variable.\n# - <ROLE>: The role description of the variable."}, {'role': 'user', 'content': 'You will give feedback to a variable with the following role: <ROLE> The system prompt </ROLE>. Here is an evaluation of the variable using a language model:\n\n<LM_SYSTEM_PROMPT> Evaluate the correctness of this sentence </LM_SYSTEM_PROMPT>\n\n<LM_INPUT> A sntence with a typo </LM_INPUT>\n\n<LM_OUTPUT> The sentence you provided does indeed contain a typo. The word "sntence" should be corrected to "sentence." </LM_OUTPUT>\n\n<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\nWe are interested in giving feedback to the The system prompt for this conversation. Specifically, give feedback to the following span of text:\n\n<VARIABLE> Evaluate the correctness of this sentence </VARIABLE>\n\nGiven the above history, describe how the The system prompt could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.\n\n'}]

# Has the gradient on the output.
# CONVERSATION_START_INSTRUCTION_CHAIN = (
#     "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
#     "Here is a conversation with a language model (LM):\n\n"
#     "{conversation}"
# )

# CONVERSATION_START_INSTRUCTION_BASE = (
#     "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
#     "Here is an evaluation of the variable using a language model:\n\n"
#     "{conversation}"
# )
CONVERSATION_START_INSTRUCTION_BASE = r"""You will give feedback to a variable with the following role:
<ROLE> {{variable_desc}} </ROLE>.
Here is an evaluation of the variable using a language model:
{{conversation_str}}
"""

CONVERSATION_START_INSTRUCTION_CHAIN = r"""You will give feedback to a variable with the following role:
<ROLE> {{variable_desc}} </ROLE>.
Here is a conversation with a language model (LM):
{{conversation_str}}
"""

# OBJECTIVE_INSTRUCTION_BASE = (
#     "<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. "
#     "Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\n"
# )

OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output.
Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>"""

# Third part of the prompt for the llm backward function.
# Asks the user to evaluate a variable in the conversation.
# EVALUATE_VARIABLE_INSTRUCTION = (
#     "We are interested in giving feedback to the {variable_desc} "
#     "for this conversation. Specifically, give feedback to the following span "
#     "of text:\n\n<VARIABLE> "
#     "{variable_short} </VARIABLE>\n\n"
#     "Given the above history, describe how the {variable_desc} "
#     "could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.\n\n"
# )

EVALUATE_VARIABLE_INSTRUCTION = r"""We are interested in giving feedback to the {{variable_desc}}
for this conversation. Specifically, give feedback to the following span of text:
<VARIABLE> {{variable_short}} </VARIABLE>
Given the above history, describe how the {{variable_desc}}
could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.
"""


BAWARD_SYSTEM_PROMPT = r"""{#Overall optimizer description#}
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
"""
# NOTE: By using f""" ... """ for the FEEDBACK_ENGINE_TEMPLATE, you format the string using Python's standard string formatting while keeping Jinja2 placeholders intact.
FEEDBACK_ENGINE_TEMPLATE = f"""<SYS>
{BAWARD_SYSTEM_PROMPT}
{GLOSSARY_TEXT_BACKWARD}
</SYS>
<USER>
{{{{conversation_sec}}}}

{{{{objective_instruction_sec}}}}

{{{{evaluate_variable_instruction_sec}}}}
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
