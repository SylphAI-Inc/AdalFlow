.. _generator:

Generator 
=========
The Center of it All 
~~~~~~~~~~~~~~~~~~~~~~~
Generator is the most essential functional component in LightRAG. 
It is a user-facing orchestration component with a simple and unified interface for LLM prediction.
It orchestrates the following components along with their required arguments:

- ``Prompt``

- ``ModelClient``

- Output processors to process the raw string response to desired format.

By switching out the model client, you can call any LLM model on your prompt, either API or local.

GeneratorOutput
^^^^^^^^^^^^^^^
Different from all other components, we can not alway enforce LLM to output the right format.
We in particular created a :class:`core.types.GeneratorOutput` (a subclass of ``DataClass``) to store `data` (parsed response), `error` (error message if either the model inference SDKs fail or the output parsing fail) and `raw_response` (raw string response for reference) for any LLM predictions.
It is in developers' hands to process the output accordingly. 

Tracing
^^^^^^^^^^^
In particular, we provide two tracing methods to help you develop and improve the Generator:

1. Trace the history change(states) on prompt during your development process. Developers typically go through a long process of prompt optimization and it is frustrating
to lose track of the prompt changes when your current change actually makes the performance much worse.

We created a `GeneratorStateLogger` to handle the logging and saving into json files. To further simplify developers's process,
we provides a class decorator `trace_generator_states` where a single line of code can be added to any of your task component. 
It will automatically track any attributes of type `Generator`.

.. code-block:: python

    from tracing import trace_generator_states
    from core import Component, Generator

    @trace_generator_states()
    class SimpleQA(Component):
        def __init__(self):
            super().__init__()
            self.generator = Generator(...)
            self.generator_2 = Generator(...)
        def call(...):

In default, a dir from the current working directory will be created to store the log files. 
The project name in defaul is `SimpleQA` and the log file will be named as `generator_state_trace.json`
where both the `generator` and `generator_2` will be logged.
The structure of log directory is as follows:

.. code-block:: bash

    .
    ├── traces
    │   ├── SimpleQA
    │   │   ├── generator_state_trace.json



Here is an example log file:

.. code-block:: json

    {
        "generator": [
            {
                "prompt_states": {
                    "_components": {},
                    "_parameters": {},
                    "training": false,
                    "_template_string": "{# task desc #}\n{% if task_desc_str %}\n{{task_desc_str}}\n{% else %}\nAnswer user query.\n{% endif %}\n{# output format #}\n{% if output_format_str %}\n<OUTPUT_FORMAT>\n{{output_format_str}}\n</OUTPUT_FORMAT>\n{% endif %}\n{# tools #}\n{% if tools_str %}\n<TOOLS>\n{{tools_str}}\n</TOOLS>\n{% endif %}\n{# example #}\n{% if examples_str %}\n<EXAMPLES>\n{{examples_str}}\n</EXAMPLES>\n{% endif %}\n{# chat history #}\n{% if chat_history_str %}\n<CHAT_HISTORY>\n{{chat_history_str}}\n</CHAT_HISTORY>\n{% endif %}\n{#contex#}\n{% if context_str %}\n<CONTEXT>\n{{context_str}}\n</CONTEXT>\n{% endif %}\n{# steps #}\n{% if steps_str %}\n<STEPS>\n{{steps_str}}\n</STEPS>\n{% endif %}\n{% if input_str %}\n<Inputs>\n{{input_str}}\n</Inputs>\n{% endif %}\n{% if output_str %}\n<Outputs>\n{{output_str}}\n</Outputs>\n{% endif %}\n",
                    "prompt_variables": [
                        "chat_history_str",
                        "context_str",
                        "examples_str",
                        "input_str",
                        "output_format_str",
                        "output_str",
                        "steps_str",
                        "task_desc_str",
                        "tools_str"
                    ],
                    "preset_prompt_kwargs": {
                        "task_desc_str": "You are a helpful assistant and with a great sense of humor."
                    }
                },
                "time_stamp": "2024-06-02T15:55:21.765794"
            },
            {
                "prompt_states": {
                    "_components": {},
                    "_parameters": {},
                    "training": false,
                    "_template_string": "{# task desc #}\n{% if task_desc_str %}\n{{task_desc_str}}\n{% else %}\nAnswer user query.\n{% endif %}\n{# output format #}\n{% if output_format_str %}\n<OUTPUT_FORMAT>\n{{output_format_str}}\n</OUTPUT_FORMAT>\n{% endif %}\n{# tools #}\n{% if tools_str %}\n<TOOLS>\n{{tools_str}}\n</TOOLS>\n{% endif %}\n{# example #}\n{% if examples_str %}\n<EXAMPLES>\n{{examples_str}}\n</EXAMPLES>\n{% endif %}\n{# chat history #}\n{% if chat_history_str %}\n<CHAT_HISTORY>\n{{chat_history_str}}\n</CHAT_HISTORY>\n{% endif %}\n{#contex#}\n{% if context_str %}\n<CONTEXT>\n{{context_str}}\n</CONTEXT>\n{% endif %}\n{# steps #}\n{% if steps_str %}\n<STEPS>\n{{steps_str}}\n</STEPS>\n{% endif %}\n{% if input_str %}\n<Inputs>\n{{input_str}}\n</Inputs>\n{% endif %}\n{% if output_str %}\n<Outputs>\n{{output_str}}\n</Outputs>\n{% endif %}\n",
                    "prompt_variables": [
                        "chat_history_str",
                        "context_str",
                        "examples_str",
                        "input_str",
                        "output_format_str",
                        "output_str",
                        "steps_str",
                        "task_desc_str",
                        "tools_str"
                    ],
                    "preset_prompt_kwargs": {
                        "task_desc_str": "You are a helpful assistant and with a great sense of humor. Second edition."
                    }
                },
                "time_stamp": "2024-06-02T15:56:37.756148"
            }
        ],
        "generator2": [
        {
            "prompt_states": {
                "_components": {},
                "_parameters": {},
                "training": false,
                "_template_string": "{# task desc #}\n{% if task_desc_str %}\n{{task_desc_str}}\n{% else %}\nAnswer user query.\n{% endif %}\n{# output format #}\n{% if output_format_str %}\n<OUTPUT_FORMAT>\n{{output_format_str}}\n</OUTPUT_FORMAT>\n{% endif %}\n{# tools #}\n{% if tools_str %}\n<TOOLS>\n{{tools_str}}\n</TOOLS>\n{% endif %}\n{# example #}\n{% if examples_str %}\n<EXAMPLES>\n{{examples_str}}\n</EXAMPLES>\n{% endif %}\n{# chat history #}\n{% if chat_history_str %}\n<CHAT_HISTORY>\n{{chat_history_str}}\n</CHAT_HISTORY>\n{% endif %}\n{#contex#}\n{% if context_str %}\n<CONTEXT>\n{{context_str}}\n</CONTEXT>\n{% endif %}\n{# steps #}\n{% if steps_str %}\n<STEPS>\n{{steps_str}}\n</STEPS>\n{% endif %}\n{% if input_str %}\n<Inputs>\n{{input_str}}\n</Inputs>\n{% endif %}\n{% if output_str %}\n<Outputs>\n{{output_str}}\n</Outputs>\n{% endif %}\n",
                "prompt_variables": [
                    "chat_history_str",
                    "context_str",
                    "examples_str",
                    "input_str",
                    "output_format_str",
                    "output_str",
                    "steps_str",
                    "task_desc_str",
                    "tools_str"
                ],
                "preset_prompt_kwargs": {
                    "task_desc_str": "You are the second generator."
                }
            },
            "time_stamp": "2024-06-03T16:44:45.223220"
        }
    ]
    }
    
2. Trace all failed LLM predictions for further improvement.

Similarly, :class:`tracing.generator_call_logger.GeneratorCallLogger` is created to log generator call input arguments and output results.
`trace_generator_call` decorator is provided to provide one-line setup to trace calls, which in default will log only failed predictions.

Adding the second decorator to the above example:

.. code-block:: python

    from tracing import trace_generator_errors

    @trace_generator_call()
    @trace_generator_states()
    class SimpleQA(Component):
        def __init__(self):
            super().__init__()
            self.generator = Generator(...)
            self.generator_2 = Generator(...)
        def call(...):

Now, three more files will be created in the log directory:

.. code-block:: bash

    .
    ├── traces
    │   ├── SimpleQA
    │   │   ├── logger_metadata.json
    │   │   ├── generator_call.jsonl
    │   │   ├── generator_2_call.jsonl

The `logger_metadata.json` file contains the metadata of the logger, it looks like this:

.. code-block:: json

    {
        "generator": "./traces/SimpleQA/generator_call.jsonl",
        "generator2": "./traces/SimpleQA/generator2_call.jsonl"
    }

The `generator_call.jsonl` file contains the log of all calls to the generator, it looks like this:

.. code-block:: json

    {"prompt_kwargs": {"input_str": "What is the capital of France?"}, "model_kwargs": {}, "output": {"data": "Bonjour!\n\nThe capital of France is Paris, of course! But did you know that the Eiffel Tower in Paris is actually the most-visited paid monument in the world? Mind-blowing, right?\n\nNow, would you like to know some more fun facts or perhaps ask another question? I'm all ears (or should I say, all eyes?)", "error_message": null, "raw_response": "Bonjour!\n\nThe capital of France is Paris, of course! But did you know that the Eiffel Tower in Paris is actually the most-visited paid monument in the world? Mind-blowing, right?\n\nNow, would you like to know some more fun facts or perhaps ask another question? I'm all ears (or should I say, all eyes?)"}, "time_stamp": "2024-06-03T16:44:45.582859"}

.. note ::

    Usually, let the evaluation run on evaluation to collect as much as failed predictions can be highly helpful for either manual prompting or auto-prompt engineering (APE).

Training [Experimental]
^^^^^^^^^^^^^^^^^^^^^^^

.. A Note on Tokenization#
.. By default, LlamaIndex uses a global tokenizer for all token counting. This defaults to cl100k from tiktoken, which is the tokenizer to match the default LLM gpt-3.5-turbo.

.. If you change the LLM, you may need to update this tokenizer to ensure accurate token counts, chunking, and prompting.

.. admonition:: API reference
   :class: highlight

   - :class:`core.generator.Generator`
   - :class:`core.types.GeneratorOutput`
   - :class:`tracing.generator_call_logger.GeneratorCallLogger`
   - :class:`tracing.generator_state_logger.GeneratorStateLogger`
