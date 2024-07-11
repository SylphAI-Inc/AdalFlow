"""This is the default system prompt template used in the LightRAG.

Use :ref:`Prompt <core-prompt_builder>` class  to manage it.
"""

__all__ = [
    "LIGHTRAG_DEFAULT_PROMPT_ARGS",
    "LIGHTRAG_DEFAULT_PROMPT_TRAINABLE_PARAMS",
    "SIMPLE_DEFAULT_LIGHTRAG_SYSTEM_PROMPT",
    "DEFAULT_LIGHTRAG_SYSTEM_PROMPT",
]
# TODO: potentially make a data class for this
LIGHTRAG_DEFAULT_PROMPT_ARGS = [
    "task_desc_str",  # task description
    "output_format_str",  # output format of the task
    "tools_str",  # tools used in the task
    "examples_str",  # examples of the task
    "chat_history_str",  # chat history of the user
    "context_str",  # context of the user query
    "steps_str",  # used in agent steps
    "input_str",  # user query or input
]

LIGHTRAG_DEFAULT_PROMPT_TRAINABLE_PARAMS = [
    "task_desc_str",
    # "output_format_str",
    "examples_str",
]

SIMPLE_DEFAULT_LIGHTRAG_SYSTEM_PROMPT = r"""<SYS>{{task_desc_str}}</SYS>
User: {{input_str}}
You:"""

DEFAULT_LIGHTRAG_SYSTEM_PROMPT = r"""<SYS>
{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
{% else %}
You are a helpful assistant.
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
{# steps #}
{% if steps_str %}
<STEPS>
{{steps_str}}
</STEPS>
{% endif %}
</SYS>
{% if input_str %}
<User>
{{input_str}}
</User>
{% endif %}
You:
"""
"""This is the default system prompt template used in the LightRAG.

Use :ref:`Prompt <core-prompt_builder>` class  to manage it.
"""
