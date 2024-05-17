class Prompt [link to api class]
============
LightRAG library in default maximize developers' control towards the final input string to the LLM model and minimize the token consumption. 
We enable advanced features without relying on API provider's prompt manipulation such as `OpenAI`'s tools or assistant APIs.

Our `Prompt` class in default takes in the following `jinjia2` template with varibles:

.. code-block:: python
   :linenos:

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

We use `<<SYS>>` and `<</SYS>>` to separate the prompt into system and user parts. Each section other than `task_desc_str` is encapulated in a special token. Different model can have different special tokens. 
Here is one example of `Llama3 Documentation <https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/>`_ prompts formatted with special tokens:

.. code-block:: 
   :linenos:

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful AI assistant for travel tips and recommendations<|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


Across our library, here our advanced features: 

- Few-shot and Many-shots In-context Learning (ICL) where the `examples_str` variable is used to pass the examples to the model.

- Tools/Function Calls where the `tools_str` variable is used to pass the tools to the model.

- Memory where the `chat_history_str` variable is used to pass the memory to the model.

- Retrieval augmented generation(RAG) where the `context_str`` variable is used to pass the retrieved context.

- Agent with multiple step planning and replanning capabilities, where the `steps_str` variable is used to pass the previous steps to the model.





