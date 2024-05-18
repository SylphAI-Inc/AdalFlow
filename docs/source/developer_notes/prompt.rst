class Prompt [link to api class]
============
LightRAG library in default maximizes developers' control towards the final experience and performance, simplify the development process, and minimize the token consumption.

For the major chat models, we eventually will only send two messages to the model: the system message and the user message. The user message is simple,
often you have a message `{'role': 'user', 'text': 'Hello, how are you?'}`. The system message is more complex, it contains the task description, tools, examples, chat history, context, and 
intermediate step history from the agent.

Our `DEFAULT_LIGHTRAG_PROMPT` decides the content you send to the system and is represented with `jinjia2` template with 6 variables: `task_desc_str`, `tools_str`, `examples_str`, `chat_history_str`, `context_str`, and `steps_str`.

.. code-block:: python
   :linenos:

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

Across our library, here our advanced features: 

- Few-shot and Many-shots In-context Learning (ICL) where the `examples_str` variable is used to pass the examples to the model.

- Tools/Function Calls where the `tools_str` variable is used to pass the tools to the model.

- Memory where the `chat_history_str` variable is used to pass the memory to the model.

- Retrieval augmented generation(RAG) where the `context_str`` variable is used to pass the retrieved context.

- Agent with multiple step planning and replanning capabilities, where the `steps_str` variable is used to pass the previous steps to the model.

Note: this means in default our out-of-box components would not support API providers's tools/function calls as we only send the system and user messages to the model.
But it should not stop you from implementing them yourself.

Prompt and Special Tokens context
---------------------


Each section other than `task_desc_str` is encapulated in a special token. Different model can have different special tokens. 
Here is one example of `Llama3 Documentation <https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/>`_ prompts formatted with special tokens:

input string to the LLM model and minimize the token consumption. 
We enable advanced features without relying on API provider's prompt manipulation such as `OpenAI`'s tools or assistant APIs.

.. code-block:: 
   :linenos:

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful AI assistant for travel tips and recommendations<|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    What can you help me with?<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>





Here is how you customize a new prompt:

.. code-block:: python
   :linenos:

    from core.prompt_builder import Prompt

    new_template = r"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {{task_desc_str}}
    Your context: {{context_str}} <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    {{query_str}}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """


    prompt = Prompt(template=new_template)



