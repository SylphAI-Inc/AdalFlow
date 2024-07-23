TEXT_LOSS_TEMPLATE = r"""<SYS>
{{sys_prompt_str}}
</SYS>
<USER>
{{user_prompt_str}}
</USER>
You:
"""

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
