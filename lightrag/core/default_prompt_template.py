LIGHTRAG_DEFAULT_PROMPT_ARGS = [
    "task_desc_str",
    "output_format_str",
    "tools_str",
    "examples_str",
    "chat_history_str",
    "context_str",
    "steps_str",
    "input_str",
    "output_str",
]
LIGHTRAG_DEFAULT_PROMPT_TRAINABLE_PARAMS = [
    "task_desc_str",
    # "output_format_str",
    "examples_str",
]

DEFAULT_LIGHTRAG_SYSTEM_PROMPT = r"""
{% if task_desc_str or output_format_str or tools_str or examples_str or chat_history_str or context_str or steps_str %}
<SYS>
{% endif %}
{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
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
{% if task_desc_str or output_format_str or tools_str or examples_str or chat_history_str or context_str or steps_str %}
</SYS>
{% endif %}
{% if input_str %}
<Inputs>
{{input_str}}
</Inputs>
{% endif %}
{% if output_str %}
<Outputs>
{{output_str}}
</Outputs>
{% endif %}
"""
"""This is the default system prompt template used in the LightRAG.

Use :ref:`Prompt<core-prompt_builder>` class  to manage it.
"""
