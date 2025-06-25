"""This is the default system prompt template used in the AdalFlow.

Use :ref:`Prompt <core-prompt_builder>` class  to manage it.
"""

__all__ = [
    "ADALFLOW_DEFAULT_PROMPT_ARGS",
    "ADALFLOW_DEFAULT_PROMPT_TRAINABLE_PARAMS",
    "SIMPLE_DEFAULT_ADALFLOW_SYSTEM_PROMPT",
    "DEFAULT_ADALFLOW_SYSTEM_PROMPT",
]
# TODO: potentially make a data class for this
ADALFLOW_DEFAULT_PROMPT_ARGS = [
    "task_desc_str",  # task description
    "output_format_str",  # output format of the task
    "tools_str",  # tools used in the task
    "examples_str",  # examples of the task
    "chat_history_str",  # chat history of the user
    "context_str",  # context of the user query
    "steps_str",  # used in agent steps
    "input_str",  # user query or input
]

ADALFLOW_DEFAULT_PROMPT_TRAINABLE_PARAMS = [
    "task_desc_str",
    # "output_format_str",
    "examples_str",
]

SIMPLE_DEFAULT_ADALFLOW_SYSTEM_PROMPT = r"""<SYS>{{task_desc_str}}</SYS>
User: {{input_str}}
You:"""

DEFAULT_ADALFLOW_SYSTEM_PROMPT = r"""<START_OF_SYSTEM_PROMPT>
{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
{% else %}
You are a helpful assistant.
{% endif %}
{#input format#}
{% if input_format_str %}
<INPUT_FORMAT>
{{input_format_str}}
</INPUT_FORMAT>
{% endif %}
{# output format #}
{% if output_format_str %}
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
{% endif %}
{# tools #}
{% if tools_str %}
<TOOLS>
{{tools_str}}
</TOOLS>
{% endif %}
{# example #}
{% if examples_str %}
<EXAMPLES>
{{examples_str}}
</EXAMPLES>
{% endif %}
{# chat history #}
{% if chat_history_str %}
<CHAT_HISTORY>
{{chat_history_str}}
</CHAT_HISTORY>
{% endif %}
{#contex#}
{% if context_str %}
<CONTEXT>
{{context_str}}
</CONTEXT>
{% endif %}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER_PROMPT>
{% if input_str %}
{{input_str}}
{% endif %}
{# steps #}
{% if steps_str %}
<START_OF_ASSISTANT_STEPS>
{{steps_str}}
<END_OF_ASSISTANT_STEPS>
{% endif %}
<END_OF_USER_PROMPT>
"""
# 1. use steps_str for agentic with multiple loop
# 2. use context_str for RAG's contex
# 3. use chat_history_str for chat history
# 4. use examples_str for examples
# 5. use tools_str for tools definition (or you can directly pass in model_kwargs as "tools")
# 6. use output_format_str for output format
# 7. use input_format_str for input format
# 8. use task_desc_str for task description

"""This is the default system prompt template used in the AdalFlow.

Use :ref:`Prompt <core-prompt_builder>` class  to manage it.
"""
