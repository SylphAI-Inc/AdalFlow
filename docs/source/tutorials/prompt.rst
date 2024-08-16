

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
      <a href="https://colab.research.google.com/drive/1_sGeHaKrwpI9RiL01g3cKyI2_5PJqZtr?usp=sharing" target="_blank" style="margin-right: 10px;">
         <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
      <a href="https://github.com/SylphAI-Inc/LightRAG/blob/main/tutorials/prompt_note.py" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

Prompt
============



.. .. admonition:: Author
..    :class: highlight

..    `Li Yin <https://github.com/liyin2015>`_


.. Context
.. ----------------

The prompt refers to the text input to the LLM models.
When sent to an LLM, the model uses the prompt to auto-regressively generate the next tokens, continuing the process until it reaches a specified stopping criterion.
The prompt itself plays a crucial role in the performance of the desired tasks.
Researchers often use `special tokens` [1]_ to separate different sections of the prompt, such as the system message, user message, and assistant message.
Ideally, developers should format this prompt with special tokens specific to the model's at training time.
However, many proprietary APIs did not disclose their special tokens, and requires users to send them in the forms of messages of different roles.

Design
----------------

`AdalFlow` seeks to maximize developers' control over the prompt.
Thus, in most cases, we help developers gather different sections and form them into one prompt.
This prompt will then be sent to the LLM as a single message.
The default role of the message we use is `system`.
Though it is not a special token, we use ``<SYS></SYS>`` to represent the system message in the prompt, which works quite well.


.. code-block:: python

    simple_prompt = r"""<SYS> You are a helpful assistant. </SYS> User: What can you help me with?"""

If it is `Llama3` model, the final text sent to the model for tokenization will be:

.. code-block:: python

   final_prompt = r"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
   {{simple_prompt}} <|eot_id|>"""

And the LLM will return the following text:

.. code-block:: python

   prediction = r"""<|start_header_id|>assistant<|end_header_id|> You can ask me anything you want. <|eot_id|><|end_of_text|>"""

Data Flow in LLM applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: /_static/images/LightRAG_dataflow.png
    :align: center
    :alt: Data Flow in LLM applications
    :width: 620px

    Data flow in LLM applications

Look at the most complicated case: We will have user query, retrieved context, task description, definition of tools, few-shot examples, past conversation history, step history from the agent, and the output format specification.
All these different parts need to be formatted into a single prompt.
We have to do all this with flexibility and also make it easy for developers to read.



Why Jinja2?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To format the prompt, you can use any of Python's native string formatting.

.. code-block:: python
   :linenos:

    # percent(%) formatting
    print("<SYS>%s</SYS> User: %s" % (task_desc_str, input_str))

    # format() method with kwargs
    print(
        "<SYS>{task_desc_str}</SYS> User: {input_str}".format(
            task_desc_str=task_desc_str, input_str=input_str
        )
    )

    # f-string
    print(f"<SYS>{task_desc_str}</SYS> User: {input_str}")

    # Templates
    from string import Template

    t = Template("<SYS>$task_desc_str</SYS> User: $input_str")
    print(t.substitute(task_desc_str=task_desc_str, input_str=input_str))


We opted for `Jinja2` [1]_ as the templating engine for the prompt.
Besides the placeholders using ``{{}}`` for keyword arguments, Jinja2 also allow users to write code similar to Python syntax.
This includes conditionals, loops, filters, and even comments, which are lacking in Python's native string formatting.
Here is one example of using `Jinja2` to format the prompt:


.. code-block:: python

   def jinja2_template_example(**kwargs):
      from jinja2 import Template

      template = r"""<SYS>{{ task_desc_str }}</SYS>
   {# tools #}
   {% if tools %}
   <TOOLS>
   {% for tool in tools %}
   {{loop.index}}. {{ tool }}
   {% endfor %}
   </TOOLS>
   {% endif %}
   User: {{ input_str }}"""
      t = Template(template, trim_blocks=True, lstrip_blocks=True)
      print(t.render(**kwargs))

Let's call it with and without tools:

.. code-block:: python

   jinja2_template_example(task_desc_str=task_desc_str, input_str=input_str)
   jinja2_template_example(
        task_desc_str=task_desc_str, input_str=input_str, tools=tools
    )

The printout would be:

.. code-block::

   <SYS>You are a helpful assitant</SYS>
   User: What is the capital of France?

And with tools:

.. code-block::

   <SYS>You are a helpful assitant</SYS>
   <TOOLS>
   1. google
   2. wikipedia
   3. wikidata
   </TOOLS>
   User: What is the capital of France?

We can see how easy and flexible to programmatically format the prompt with `Jinja2`.



Prompt class
----------------


We created our :class:`Prompt Component<core.prompt_builder.Prompt>` to render the prompt with the string ``template`` and ``prompt_kwargs``.
It is a simple component, but it is quite handy.
Let's use the same template as above:

.. code-block:: python

   from adalflow.core.prompt_builder import Prompt

   prompt = Prompt(
      template=template,
      prompt_kwargs={
         "task_desc_str": task_desc_str,
         "tools": tools,
      },
   )
   print(prompt)
   print(prompt(input_str=input_str)) # takes the rest arguments in keyword arguments

The ``Prompt`` class allow us to preset some of the prompt arguments at initialization, and then we can call the prompt with the rest of the arguments.
Also, by subclassing ``Component``, we can easily visualize this component with ``print``.
Here is the output:

.. code-block::

   Prompt(
      template: <SYS>{{ task_desc_str }}</SYS>
      {# tools #}
      {% if tools %}
      <TOOLS>
      {% for tool in tools %}
      {{loop.index}}. {{ tool }}
      {% endfor %}
      </TOOLS>
      {% endif %}
      User: {{ input_str }}, prompt_kwargs: {'task_desc_str': 'You are a helpful assitant', 'tools': ['google', 'wikipedia', 'wikidata']}, prompt_variables: ['input_str', 'tools', 'task_desc_str']
   )

As with all components, you can use ``to_dict`` and ``from_dict`` to serialize and deserialize the component.

Default Prompt Template
-------------------------

In default, the ``Prompt`` class uses the :const:`DEFAULT_LIGHTRAG_SYSTEM_PROMPT<core.default_prompt_template.DEFAULT_LIGHTRAG_SYSTEM_PROMPT>` as its string template if no template is provided.
This default template allows you to conditionally passing seven important variables designed from the data flow diagram above.
These varaibles are:

.. code-block:: python

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

Now, let's see the minimum case where we only have the user query:

.. code-block:: python

   prompt = Prompt()
   output = prompt(input_str=input_str)
   print(output)

The output will be the bare minimum with only the user query and a prefix for assistant to respond:

.. code-block::

   <User>
   What is the capital of France?
   </User>
   You:

.. note::

   In reality, we barely need to use the raw ``Prompt`` class directly as it is orchestrated by the ``Generator`` component together with the ``ModelClient`` that we will introduce next.




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
   - :const:`core.default_prompt_template.DEFAULT_LIGHTRAG_SYSTEM_PROMPT`
