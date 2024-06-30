Prompt
============
.. admonition:: Author
   :class: highlight

   `Li Yin <https://github.com/liyin2015>`_

We strick to maximize developers' control towards the final experience and performance, simplify the development process, and minimize the token consumption.

For the major chat models, we eventually will only send two messages to the model: the system message and the user message. The user message is simple,
often you have a message `{'role': 'user', 'content': 'Hello, how are you?'}`. The system message is more complex, it contains the task description, tools, examples, chat history, context, and
intermediate step history from agents.

Prompt template
---------------------

Our `DEFAULT_LIGHTRAG_SYSTEM_PROMPT` templates the system prompt with 7 important sections. We leverage `jinjia2` template for **programmable prompt** right along with string.

The default template comes  with 7 variables: `task_desc_str`, `output_format_str`, `tools_str`, `examples_str`, `chat_history_str`, `context_str`, and `steps_str`.

A jinjia2 template will rendered with :ref:`Prompt<core-prompt_builder>` class. If some fields being empty, that section will be empty in the final prompt string.

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

- Various output formats where the `output_format_str` variable is used to pass the output format to the model.

- Few-shot and Many-shots In-context Learning (ICL) where the `examples_str` variable is used to pass the examples to the model.

- Tools/Function Calls where the `tools_str` variable is used to pass the tools to the model.

- Memory where the `chat_history_str` variable is used to pass the memory to the model.

- Retrieval augmented generation(RAG) where the `context_str`` variable is used to pass the retrieved context.

- Agent with multiple step planning and replanning capabilities, where the `steps_str` variable is used to pass the previous steps to the model.

**Note: this means in default our out-of-box components would not support API providers's tools/function calls as we only send the system and user messages to the model.
But it should not stop you from implementing them yourself.**

Prompt class
---------------------
We designed a :ref:`Prompt<core-prompt_builder>` class  to render the `template` with the variables to string as the final system prompt. In the simplest case, the string is empty and we will only send
a user message to the model. And in most cases, you want to add at least the `task_desc_str` to the system message.

The cool thing about our `Prompt` system is how flexible it can be. If you need to put another `template` for say `task_desc_str`, you can do that using the `Prompt` class.
For example, your task is to instruct the llm to choose `top_k` from the given choices, you can define a new template like this:

.. code-block:: python
   :linenos:

   from core.prompt_builder import Prompt

   task_desc_template = r"""
   Choose the top {{top_k}} from the following choices: {{choices}}
   """
   top_k = 3
   choices = ['apple', 'banana', 'orange', 'grape']
   task_desc_prompt = Prompt(template=task_desc_template, preset_prompt_kwargs={'top_k': top_k, 'choices': choices})
   task_desc_str = task_desc_prompt.call()
   prompt = Prompt(preset_prompt_kwargs={'task_desc_str': task_desc_str})
   prompt.print_prompt()

The output would be:

.. code-block:: xml
   :linenos:

   Choose the top 3 from the following choices: ['apple', 'banana', 'orange', 'grape']




Prompt and Special Tokens context
----------------------------------


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


Prompt Engineering experience
-------------------------------
There is not robust prompt, and it is one of the most sensitive creatures in the AI world.
Here are some tips:
- Even the output format matters, the order of your output fields, the formating.
Output yaml or json format can lead to different performance. We have better luck with yaml format.
- Few-shot works so well in some case, but it can lead to regression in some cases.
- It is not fun to be a prompt engineer! But what can we do for now.


Resources:
1. `Jinja2`:
