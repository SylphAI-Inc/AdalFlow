LIGHTRAG_DEFAULT_PROMPT_ARGS = [
    "task_desc_str",
    "output_format_str",
    "tools_str",
    "examples_str",
    "chat_history_str",
    "context_str",
    "steps_str",
]

DEFAULT_LIGHTRAG_SYSTEM_PROMPT = r"""{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
{% else %}
Answer user query.
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
"""
"""This is the default system prompt template used in the LightRAG.

Use :ref:`Prompt<core-prompt_builder>` class  to manage it.
"""
