DEFAULT_LIGHTRAG_SYSTEM_PROMPT = r"""{# task desc #}
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
