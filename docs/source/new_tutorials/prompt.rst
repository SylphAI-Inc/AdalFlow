

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 20px;">
      <a href="https://colab.research.google.com/drive/1_sGeHaKrwpI9RiL01g3cKyI2_5PJqZtr?usp=sharing" target="_blank" style="margin-right: 10px;">
         <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
      <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/prompt_note.py" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

Prompt
============

AdalFlow leverages `Jinja2` [1]_  to programmatically format the prompt for the language model.
By aggregating different parts of the prompt, it is 10X easier to understand the LLM applications.
We created a :class:`Prompt <core.prompt_builder.Prompt>` class to allow developers to render the prompt with the string ``template`` and ``prompt_kwargs`` conveniently and securely.

Introduction
----------------
A `prompt` is the input text given to a language model(LM) to generate responses.
We believe in `prompting` is the new programming language and developers need to seek maximum control over the prompt.


.. figure:: /_static/images/LightRAG_dataflow.png
    :align: center
    :alt: Data Flow in LLM applications
    :width: 620px

    Data flow in LLM applications

Various LLM app patterns, from RAG to agents, can be implemented via formatting a subpart of the prompt.

Researchers often use `special tokens` [2]_ to separate different sections of the prompt, such as the system message, user message, and assistant message.
If it is `Llama3` model, the final text sent to the model for tokenization will be:

.. code-block:: python

   final_prompt = r"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
   {{simple_prompt}} <|eot_id|>"""

And the LLM will return the following text:

.. code-block:: python

   prediction = r"""<|start_header_id|>assistant<|end_header_id|> You can ask me anything you want. <|eot_id|><|end_of_text|>"""

However, many proprietary APIs did not disclose their special tokens, and requires users to send them in the forms of messages of different roles.



Use Prompt Class
----------------
Besides the placeholders using ``{{}}`` for keyword arguments, Jinja2 also allow users to write code similar to Python syntax.
This includes conditionals, loops, filters, and even comments, which are lacking in Python's native string formatting.

In default, the ``Prompt`` class uses the :const:`DEFAULT_ADALFLOW_SYSTEM_PROMPT<core.default_prompt_template.DEFAULT_ADALFLOW_SYSTEM_PROMPT>` as its string template if no template is provided.
But it is super easy to create your own template with Jinja2 syntax.
Here is one example of using `Jinja2` to format the prompt with comments `{# #}` and code blocks `{% %}`:


.. code-block:: python

   import adalflow as adal

   template = r"""<START_OF_SYSTEM_MESSAGE>{{ task_desc_str }}<END_OF_SYSTEM_MESSAGE>
   {# tools #}
   {% if tools %}
   <TOOLS>
   {% for tool in tools %}
   {{loop.index}}. {{ tool }}
   {% endfor %}
   </TOOLS>{% endif %}
   <START_OF_USER>{{ input_str }} <END_OF_USER>"""

    task_desc_str = "You are a helpful assitant"

    tools = ["google", "wikipedia", "wikidata"]

    prompt = adal.Prompt(
        template=template,
        prompt_kwargs={
            "task_desc_str": task_desc_str,
            "tools": tools,
        },
    )

   print(prompt(input_str="What is the capital of France?"))

The printout would be:

.. code-block::

   <START_OF_SYSTEM_MESSAGE>You are a helpful assitant<END_OF_SYSTEM_MESSAGE>
   <TOOLS>
   1. google
   2. wikipedia
   3. wikidata
   </TOOLS>
   <START_OF_USER>What is the capital of France? <END_OF_USER>

As with all components, you can use ``to_dict`` and ``from_dict`` to serialize and deserialize the component.


.. note::

   In reality, we barely need to use the raw ``Prompt`` class directly as it is orchestrated by the ``Generator``.

You do not need to worry about handling all functionalities of a prompt, (1) we have `Parser` such as `JsonParser`, `DataClassParser` to help you handle the outpt formatting,
(2) we `FuncTool` to help you describe a functional tool in the prompt.


.. Prompt Engineering experience
.. -------------------------------
.. There is no robust prompt, and it is one of the most sensitive creatures in the AI world.
.. Here are some tips:

.. - Even the output format matters, the order of your output fields, the formating. Output yaml or json format can lead to different performance. We have better luck with yaml format.
.. - Few-shot works so well in some case, but it can lead to regression in some cases.
.. - It is not fun to be a prompt engineer! But what can we do for now.

.. admonition:: References
   :class: highlight

   .. [1] Jinja2: https://jinja.palletsprojects.com/en/3.1.x/
   .. [2] Llama3 special tokens: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

.. admonition:: API References
   :class: highlight

   - :class:`core.prompt_builder.Prompt`
   - :const:`core.default_prompt_template.DEFAULT_ADALFLOW_SYSTEM_PROMPT`
