Prompt
============
LightRAG library in default maximize developers' control towards the final input string to the LLM model. Our default prompt is a `jinjia2` template with varibles supporting the following features:
- Few-shot and Many-shots In-context Learning (ICL) where the `examples_str` variable is used to pass the examples to the model.
- Tools/Function Calls where the 'tools_str' variable is used to pass the tools to the model.
- Memory where the 'memory_str' variable is used to pass the memory to the model.
- Retrieval augmented generation(RAG) where the 'context_str' variable is used to pass the retrieved context.
- Agent with multiple step planning and replanning capabilities, where the `steps_str` variable is used to pass the previous steps to the model.

```
DEFAULT_LIGHTRAG_PROMPT = r"""
<<SYS>>{# task desc #}
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
```
