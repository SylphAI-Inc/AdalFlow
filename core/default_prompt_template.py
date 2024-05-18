DEFAULT_LIGHTRAG_PROMPT = r"""<<SYS>>{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
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
<</SYS>>
---------------------
{# chat history #}
{% if chat_history_str %}
<CHAT_HISTORY>
{{chat_history_str}}
</CHAT_HISTORY>
{% endif %}
User query: {{query_str}}
{#contex#}
{% if context_str %}
<CONTEXT>
{{context_str}}
</CONTEXT>
{% endif %}
{# steps #}
{% if steps_str %}
{{steps_str}}
{% endif %}
{# assistant response #}
You:
"""
