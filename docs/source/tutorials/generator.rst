.. _generator:

Generator
=========

.. .. admonition:: Author
..    :class: highlight

..    `Li Yin <https://github.com/liyin2015>`_

.. *The Center of it All*


`Generator` is a user-facing orchestration component with a simple and unified interface for LLM prediction.
It is a pipeline consisting of three subcomponents.

Design
---------------------------------------

.. figure:: /_static/images/generator.png
    :align: center
    :alt: LightRAG generator design
    :width: 700px

    Generator - The Orchestrator for LLM Prediction




The :class:`Generator<core.generator.Generator>` is designed to achieve the following goals:

1. Model Agnostic: The Generator should be able to call any LLM model with the same prompt.
2. Unified Interface: It should manage the pipeline from prompt(input)->model call -> output parsing.
3. Unified Output: This will make it easy to log and save records of all LLM predictions.
4. Work with Optimizer: It should be able to work with Optimizer to optimize the prompt.

The first three goals apply to other orchestrator components like :ref:`Retriever<tutorials-retriever>`, :ref:`Embedder<tutorials-embedder>`, and :ref:`Agent<tutorials-agent>` (mostly) as well.


An Orchestrator
^^^^^^^^^^^^^^^^^

It orchestrates three components:

- `Prompt`: by taking in ``template`` (string) and ``prompt_kwargs`` (dict) to format the prompt at initialization.
  When the ``template`` is not provided, it defaults to :const:`DEFAULT_LIGHTRAG_SYSTEM_PROMPT<core.default_prompt_template.DEFAULT_LIGHTRAG_SYSTEM_PROMPT>`.

- `ModelClient`: by taking in an already instantiated ``model_client`` and ``model_kwargs`` to call the model.
  Switching out the model client allows you to call any LLM model using the same prompt and output parsing.

- `output_processors`: A single component or chained components via :class:`Sequential<core.container.Sequential>` to process the raw response to desired format.
  If no output processor provided, it is decided by the model client and often returns raw string response (from the first response message).

**Call and arguments**

The `Generator` supports both the ``call`` (``__call__``) and ``acall`` methods.
They take two optional arguments:

- ``prompt_kwargs`` (dict): This is combined with the ``prompt_kwargs`` from the initial ``Prompt`` component and used to format the prompt.
- ``model_kwargs`` (dict): This is  combined with the ``model_kwargs`` from the initial model client, and along with :const:`ModelType.LLM<core.types.ModelType.LLM>`, it is passed to the ``ModelClient``.
  The ModelClient will interpret all the inputs as ``api_kwargs`` specific to each model API provider.



.. note ::

    This also means any ``ModelClient`` that wants to be compatible with `Generator` should take accept ``model_kwargs`` and ``model_type`` as arguments.






GeneratorOutput
^^^^^^^^^^^^^^^^^
Unlike other components, we cannot always enforce the LLM to follow the output format. The `ModelClient` and the `output_processors` may fail.


.. note::
    Whenever an error occurs, we do not raise the error and force the program to stop.
    Instead, `Generator` will always return an output record.
    We made this design choice because it can be really helpful to log various failed cases in your train/eval sets all together for further investigation and improvement.



In particular, we created :class:`GeneratorOutput<core.types.GeneratorOutput>` to capture important information.

- `data` (object) : Stores the final processed response after all three components in the pipeline, indicating `success`.
- `error` (str): Contains the error message if any of the three components in the pipeline fail. When this is not `None`, it indicates `failure`.
- `raw_response` (str): Raw string response for reference of any LLM predictions. Currently, it is a string that comes from the first response message. [This might change and be different in the future]
- `metadata` (dict): Stores any additional information
- `usage`:  Reserved for tracking the usage of the LLM prediction.

Whether to do further processing or terminate the pipeline whenever an error occurs is up to the user from here on.



Generator In Action
---------------------------------------

We will create a simple one-turn chatbot to demonstrate how to use the Generator.

Minimum Example
^^^^^^^^^^^^^^^^^

The minimum setup to initiate a generator in the code:

.. code-block:: python

    from lightrag.core import Generator
    from lightrag.components.model_client import GroqAPIClient

    generator = Generator(
        model_client=GroqAPIClient(),
        model_kwargs={"model": "llama3-8b-8192"},
    )
    print(generator)

The structure of generator using ``print``:

.. raw:: html

    <div style="max-height: 300px; overflow-y: auto;">
        <pre>
            <code class="language-python">
        Generator(
        model_kwargs={'model': 'llama3-8b-8192'},
        (prompt): Prompt(
            template: <SYS>
            {# task desc #}
            {% if task_desc_str %}
            {{task_desc_str}}
            {% else %}
            You are a helpful assistant.
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
            </SYS>
            {% if input_str %}
            <User>
            {{input_str}}
            </User>
            {% endif %}
            You:
            , prompt_variables: ['input_str', 'tools_str', 'context_str', 'steps_str', 'task_desc_str', 'chat_history_str', 'output_format_str', 'examples_str']
        )
        (model_client): GroqAPIClient()
        )
            </code>
        </pre>
    </div>

**Show the Final Prompt**


The `Generator` 's ``print_prompt`` method will simply relay the method from the `Prompt` component:

.. code-block:: python

    prompt_kwargs = {"input_str": "What is LLM? Explain in one sentence."}
    generator.print_prompt(**prompt_kwargs)

The output will be the formatted prompt:

.. code-block::

    <User>
    What is LLM? Explain in one sentence.
    </User>
    You:



**Call the Generator**

.. code-block:: python

    output = generator(
        prompt_kwargs=prompt_kwargs,
    )
    print(output)

The output will be the `GeneratorOutput` object:

.. code-block::

    GeneratorOutput(data='LLM stands for Large Language Model, a type of artificial intelligence that is trained on vast amounts of text data to generate human-like language outputs, such as conversations, text, or summaries.', error=None, usage=None, raw_response='LLM stands for Large Language Model, a type of artificial intelligence that is trained on vast amounts of text data to generate human-like language outputs, such as conversations, text, or summaries.', metadata=None)

Use Template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we will use a customized template to format the prompt.
We intialized the prompt with one variable `task_desc_str`, which is further combined with the `input_str` in the prompt.

.. code-block:: python

    template = r"""<SYS>{{task_desc_str}}</SYS>
    User: {{input_str}}
    You:"""
    generator = Generator(
        model_client=GroqAPIClient(),
        model_kwargs={"model": "llama3-8b-8192"},
        template=template,
        prompt_kwargs={"task_desc_str": "You are a helpful assistant"},
    )

    prompt_kwargs = {"input_str": "What is LLM?"}

    generator.print_prompt(
        **prompt_kwargs,
    )
    output = generator(
        prompt_kwargs=prompt_kwargs,
    )

The final prompt is:

.. code-block::

    <SYS>You are a helpful assistant</SYS>
    User: What is LLM?
    You:

.. note::

    It is quite straightforward to use any prompt.
    They only need to stick to ``jinja2`` syntax.


Use output_processors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we will instruct the LLM to output a JSON object in response.
We will use the `JsonParser` to parse the output back to a `dict` object.


.. code-block:: python

    from lightrag.core import Generator
    from lightrag.core.types import GeneratorOutput
    from lightrag.components.model_client import OpenAIClient
    from lightrag.core.string_parser import JsonParser

    output_format_str = r"""Your output should be formatted as a standard JSON object with two keys:
    {
        "explaination": "A brief explaination of the concept in one sentence.",
        "example": "An example of the concept in a sentence."
    }
    """

    generator = Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo"},
        prompt_kwargs={"output_format_str": output_format_str},
        output_processors=JsonParser(),
    )

    prompt_kwargs = {"input_str": "What is LLM?"}
    generator.print_prompt(**prompt_kwargs)

    output: GeneratorOutput = generator(prompt_kwargs=prompt_kwargs)
    print(type(output.data))
    print(output.data)

The final prompt is:

.. code-block::


    <SYS>
    <OUTPUT_FORMAT>
    Your output should be formatted as a standard JSON object with two keys:
        {
            "explaination": "A brief explaination of the concept in one sentence.",
            "example": "An example of the concept in a sentence."
        }

    </OUTPUT_FORMAT>
    </SYS>
    <User>
    What is LLM?
    </User>
    You:

The above printout is:

.. code-block::

    <class 'dict'>
    {'explaination': 'LLM stands for Large Language Model, which are deep learning models trained on enormous amounts of text data.', 'example': 'An example of a LLM is GPT-3, which can generate human-like text based on the input provided.'}

Please refer to :doc:`output_parsers` for a more comprehensive guide on the `Parser` components.

Switch the model_client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Also, did you notice that we have already switched to using models from `OpenAI` in the above example?
This demonstrates how easy it is to switch the `model_client` in the Generator, making it a truly model-agnostic component.
We can even use :class:`ModelClientType<core.types.ModelClientType>` to switch the model client without handling multiple imports.

.. code-block:: python

    from lightrag.core.types import ModelClientType

    generator = Generator(
        model_client=ModelClientType.OPENAI(),  # or ModelClientType.GROQ()
        model_kwargs={"model": "gpt-3.5-turbo"},
    )

Get Errors in GeneratorOutput
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will use an incorrect API key to delibrately create an error.
We will still get a response, but it will only contain empty ``data`` and an error message.
Here is an example of an API key error with OpenAI:


.. code-block:: python

    GeneratorOutput(data=None, error="Error code: 401 - {'error': {'message': 'Incorrect API key provided: ab. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}", usage=None, raw_response=None, metadata=None)


Create from Configs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As with all components, we can create the generator purely from configs.


**Know it is a Generator**

In this case, we know we are creating a generator, we will use ``from_config`` method from the ``Generator`` class.

.. code-block:: python

    from lightrag.core import Generator

    config = {
        "model_client": {
            "component_name": "GroqAPIClient",
            "component_config": {},
        },
        "model_kwargs": {
            "model": "llama3-8b-8192",
        },
    }

    generator: Generator = Generator.from_config(config)
    print(generator)

    prompt_kwargs = {"input_str": "What is LLM? Explain in one sentence."}
    generator.print_prompt(**prompt_kwargs)
    output = generator(
        prompt_kwargs=prompt_kwargs,
    )
    print(output)


**Purely from the Configs**

This is even more general.
This method can be used to create any component from configs.
We just need to follow the config structure: ``component_name`` and ``component_config`` for all arguments.



.. code-block:: python

    from lightrag.utils.config import new_component
    from lightrag.core import Generator

    config = {
        "generator": {
            "component_name": "Generator",
            "component_config": {
                "model_client": {
                    "component_name": "GroqAPIClient",
                    "component_config": {},
                },
                "model_kwargs": {
                    "model": "llama3-8b-8192",
                },
            },
        }
    }

    generator: Generator = new_component(config["generator"])
    print(generator)

    prompt_kwargs = {"input_str": "What is LLM? Explain in one sentence."}
    generator.print_prompt(**prompt_kwargs)
    output = generator(
        prompt_kwargs=prompt_kwargs,
    )
    print(output)

It works exactly the same as the previous example.
We imported ``Generator`` in this case to only show the type hinting.

.. note::

    Please refer to the :doc:`configurations<configs>` for more details on how to create components from configs.


Examples Across the Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides these examples, LLM is like water, even in our library, we have components that have adpated Generator to various other functionalities.

- :class:`LLMRetriever<components.retriever.llm_retriever.LLMRetriever>` is a retriever that uses Generator to call LLM to retrieve the most relevant documents.
- :class:`DefaultLLMJudge<eval.llm_as_judge.DefaultLLMJudge>` is a judge that uses Generator to call LLM to evaluate the quality of the response.
- :class:`LLMOptimizer<optim.llm_optimizer.LLMOptimizer>` is an optimizer that uses Generator to call LLM to optimize the prompt.

Tracing
---------------------------------------



In particular, we provide two tracing methods to help you develop and improve the ``Generator``:

1. Trace the history change (states) on prompt during your development process.
2. Trace all failed LLM predictions for further improvement.

As this note is getting rather long. Please refer to the :doc:`tracing<logging_tracing>` to learn about these two tracing methods.


Training [Experimental]
---------------------------------------
Coming soon!

.. A Note on Tokenization#
.. By default, LlamaIndex uses a global tokenizer for all token counting. This defaults to cl100k from tiktoken, which is the tokenizer to match the default LLM gpt-3.5-turbo.

.. If you change the LLM, you may need to update this tokenizer to ensure accurate token counts, chunking, and prompting.

.. admonition:: API reference
   :class: highlight

   - :class:`core.generator.Generator`
   - :class:`core.types.GeneratorOutput`
   - :class:`core.default_prompt_template.DEFAULT_LIGHTRAG_SYSTEM_PROMPT`
   - :class:`core.types.ModelClientType`
   - :class:`core.types.ModelType`
   - :class:`core.string_parser.JsonParser`
   - :class:`core.prompt_builder.Prompt`
   - :class:`tracing.generator_call_logger.GeneratorCallLogger`
   - :class:`tracing.generator_state_logger.GeneratorStateLogger`
   - :class:`components.retriever.llm_retriever.LLMRetriever`
   - :class:`eval.llm_as_judge.DefaultLLMJudge`
   - :class:`optim.llm_optimizer.LLMOptimizer`
   - :func:`utils.config.new_component`
