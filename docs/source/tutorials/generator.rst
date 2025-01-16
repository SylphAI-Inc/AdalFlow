.. _generator:

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
      <a href="https://colab.research.google.com/drive/1gmxeX1UuUxZDouWhkLGQYrD4hAdt9IVX?usp=sharing" target="_blank" style="margin-right: 10px;">
         <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
      <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/generator_note.py" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>


Generator
=========

.. .. admonition:: Author
..    :class: highlight

..    `Li Yin <https://github.com/liyin2015>`_

.. *The Center of it All*


`Generator` is a user-facing orchestration component with a simple and unified interface for LLM prediction.
It is a pipeline consisting of three subcomponents. By switching the prompt template, model client, and output parser, users have full control and flexibility.

Design
---------------------------------------

.. figure:: /_static/images/generator.png
    :align: center
    :alt: AdalFlow generator design
    :width: 700px

    Generator - The Orchestrator for LLM Prediction



The :class:`Generator<core.generator.Generator>` is designed to achieve the following goals:

1. Model Agnostic: The Generator should be able to call any LLM model with the same prompt.
2. Unified interface: It manages the pipeline from prompt (input) -> model call -> output parsing, while still giving users full control over each part.
3. Unified Output: This will make it easy to log and save records of all LLM predictions.
4. Work with Optimizer: It should be able to work with Optimizer to optimize the prompt.

The first three goals apply to other orchestrator components like :ref:`Retriever<tutorials-retriever>`, :ref:`Embedder<tutorials-embedder>`, and :ref:`Agent<tutorials-agent>` (mostly) as well.


An Orchestrator
^^^^^^^^^^^^^^^^^

It orchestrates three components:

- `Prompt`: by taking in ``template`` (string) and ``prompt_kwargs`` (dict) to format the prompt at initialization.
  When the ``template`` is not provided, it defaults to :const:`DEFAULT_ADALFLOW_SYSTEM_PROMPT<core.default_prompt_template.DEFAULT_ADALFLOW_SYSTEM_PROMPT>`.

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


Basic Generator Tutorial
=====================

The Generator class is the core component in AdalFlow for interacting with AI models. This tutorial covers the essential concepts and patterns.

What is a Generator?
------------------

A Generator is a unified interface for model interactions that:

1. Takes input and formats it using a prompt template
2. Sends the formatted input to an AI model
3. Returns a standardized ``GeneratorOutput`` object

Basic Usage
----------

Here's the simplest way to use a Generator:

.. code-block:: python

    from adalflow.core import Generator
    from adalflow.components.model_client.openai_client import OpenAIClient

    # Create a generator
    gen = Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o-mini",
            "temperature": 0.7
        }
    )

    # Use the generator
    response = gen({"input_str": "What is the capital of France?"})
    print(response.raw_response)

Understanding the Output
----------------------

Every Generator call returns a ``GeneratorOutput`` object:

.. code-block:: python

    response = gen({"input_str": "Hello"})
    
    # Access different parts of the response
    print(response.raw_response)  # Raw model output
    print(response.data)          # Processed data (if using output processors)
    print(response.error)         # Error message if something went wrong
    print(response.usage)         # Token usage information

When to Create a Subclass
-----------------------

You should create a Generator subclass in two main cases:

1. **Different Model Types**: When using non-LLM endpoints

   .. code-block:: python

    class ImageGenerator(Generator):
        """For DALL-E and other image generation models"""
        model_type = ModelType.IMAGE_GENERATION

2. **Custom Processing**: When you need special input/output handling

   .. code-block:: python

    class CustomGenerator(Generator):
        def _pre_call(self, prompt_kwargs, model_kwargs):
            # Custom preprocessing
            return super()._pre_call(prompt_kwargs, model_kwargs)

When NOT to Subclass
------------------

Don't create a subclass for:

1. **Model Parameters**: Use ``model_kwargs`` instead

   .. code-block:: python

    # Just pass parameters directly
    gen = Generator(
        model_client=client,
        model_kwargs={
            "model": "gpt-4o-mini",
            "temperature": 0.9
        }
    )

2. **Output Processing**: Use output processors

   .. code-block:: python

    from adalflow.components.output_processors import JsonParser

    gen = Generator(
        model_client=client,
        output_processors=JsonParser()  # Process output as JSON
    )

Common Patterns
-------------

1. **Error Handling**:

   .. code-block:: python

    response = gen({"input_str": "Query"})
    if response.error:
        print(f"Error: {response.error}")
    else:
        print(response.raw_response)

2. **Async Usage**:

   .. code-block:: python

    async def generate():
        response = await gen.acall({"input_str": "Hello"})
        print(response.raw_response)

3. **Streaming**:

   .. code-block:: python

    gen = Generator(
        model_client=client,
        model_kwargs={"stream": True}
    )
    for chunk in gen({"input_str": "Tell me a story"}):
        print(chunk)

Model Types
----------

Generator supports different model types through ``ModelType``:

- ``ModelType.LLM``: Text generation (default)
- ``ModelType.IMAGE_GENERATION``: Image generation (DALL-E)
- ``ModelType.EMBEDDER``: Text embeddings
- ``ModelType.RERANKER``: Document reranking

Best Practices
------------

1. Always check for errors in the response
2. Use output processors for structured outputs
3. Set model parameters in ``model_kwargs``
4. Use async methods for better performance in async contexts
5. Use streaming for long responses

Remember: The Generator is designed to provide a consistent interface regardless of the underlying model or task. 

Generator In Action
---------------------------------------

We will create a simple one-turn chatbot to demonstrate how to use the Generator.

Minimum Example
^^^^^^^^^^^^^^^^^

The minimum setup to initiate a generator in the code:

.. code-block:: python

    import adalflow as adal
    from adalflow.components.model_client import GroqAPIClient

    generator = adal.Generator(
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

    from adalflow.core import Generator
    from adalflow.core.types import GeneratorOutput
    from adalflow.components.model_client import OpenAIClient
    from adalflow.core.string_parser import JsonParser

    output_format_str = r"""Your output should be formatted as a standard JSON object with two keys:
    {
        "explanation": "A brief explanation of the concept in one sentence.",
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
            "explanation": "A brief explanation of the concept in one sentence.",
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
    {'explanation': 'LLM stands for Large Language Model, which are deep learning models trained on enormous amounts of text data.', 'example': 'An example of a LLM is GPT-3, which can generate human-like text based on the input provided.'}

Please refer to :doc:`output_parsers` for a more comprehensive guide on the `Parser` components.

Switch the model_client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Also, did you notice that we have already switched to using models from `OpenAI` in the above example?
This demonstrates how easy it is to switch the `model_client` in the Generator, making it a truly model-agnostic component.
We can even use :class:`ModelClientType<core.types.ModelClientType>` to switch the model client without handling multiple imports.

.. code-block:: python

    from adalflow.core.types import ModelClientType

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

    from adalflow.core import Generator

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

    from adalflow.utils.config import new_component
    from adalflow.core import Generator

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
- :class:`TGDOptimizer<optim.text_grad.tgd_optimizer.TGDOptimizer>` is an optimizer that uses Generator to call LLM to optimize the prompt.
- :class:`ReAct Agent Planner<components.agent.react.ReActAgent>` is an LLM planner that uses Generator to plan and to call functions in ReAct Agent.

Tracing
---------------------------------------



In particular, we provide two tracing methods to help you develop and improve the ``Generator``:

1. Trace the history change (states) on prompt during your development process.
2. Trace all failed LLM predictions for further improvement.

As this note is getting rather long. Please refer to the :doc:`tracing<logging_tracing>` to learn about these two tracing methods.


Training
---------------------------------------
Generator in default support training mode.
It will require users to define ``Parameter`` and pass it to the ``prompt_kwargs``.

.. A Note on Tokenization#
.. By default, LlamaIndex uses a global tokenizer for all token counting. This defaults to cl100k from tiktoken, which is the tokenizer to match the default LLM gpt-3.5-turbo.

.. If you change the LLM, you may need to update this tokenizer to ensure accurate token counts, chunking, and prompting.

Image Generation
-------------------------------------------------

The Generator class also supports image generation through DALL-E models. First, you need to define a Generator subclass with the correct model type:

.. code-block:: python

    from adalflow import Generator
    from adalflow.core.types import ModelType

    class ImageGenerator(Generator):
        """Generator subclass for image generation."""
        model_type = ModelType.IMAGE_GENERATION

Then you can use it like this:

.. code-block:: python

    from adalflow import OpenAIClient

    generator = ImageGenerator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "dall-e-3",  # or "dall-e-2"
            "size": "1024x1024",  # "1024x1024", "1024x1792", or "1792x1024" for DALL-E 3
            "quality": "standard",  # "standard" or "hd" (DALL-E 3 only)
            "n": 1  # Number of images (1 for DALL-E 3, 1-10 for DALL-E 2)
        }
    )

    # Generate an image from text
    response = generator(
        prompt_kwargs={"input_str": "A white siamese cat in a space suit"}
    )
    # response.data will contain the image URL

    # Edit an existing image
    response = generator(
        prompt_kwargs={"input_str": "Add a red hat"},
        model_kwargs={
            "model": "dall-e-2",
            "image": "path/to/cat.png",  # Original image
            "mask": "path/to/mask.png"   # Optional mask showing where to edit
        }
    )

    # Create variations of an image
    response = generator(
        prompt_kwargs={"input_str": None},  # Not needed for variations
        model_kwargs={
            "model": "dall-e-2",
            "image": "path/to/cat.png"  # Image to create variations of
        }
    )

The generator supports:

- Image generation from text descriptions using DALL-E 3 or DALL-E 2
- Image editing with optional masking (DALL-E 2)
- Creating variations of existing images (DALL-E 2)
- Both local file paths and base64-encoded images
- Various image sizes and quality settings
- Multiple output formats (URL or base64)

The response will always be wrapped in a ``GeneratorOutput`` object, maintaining consistency with other AdalFlow operations. The generated image(s) will be available in the ``data`` field as either a URL or base64 string.

.. admonition:: API reference
   :class: highlight

   - :class:`core.generator.Generator`
   - :class:`core.types.GeneratorOutput`
   - :class:`core.default_prompt_template.DEFAULT_ADALFLOW_SYSTEM_PROMPT`
   - :class:`core.types.ModelClientType`
   - :class:`core.types.ModelType`
   - :class:`core.string_parser.JsonParser`
   - :class:`core.prompt_builder.Prompt`
   - :class:`tracing.generator_call_logger.GeneratorCallLogger`
   - :class:`tracing.generator_state_logger.GeneratorStateLogger`
   - :class:`components.retriever.llm_retriever.LLMRetriever`
   - :class:`components.agent.react.ReActAgent`
   - :class:`eval.llm_as_judge.DefaultLLMJudge`
   - :class:`optim.text_grad.tgd_optimizer.TGDOptimizer`
   - :func:`utils.config.new_component`
