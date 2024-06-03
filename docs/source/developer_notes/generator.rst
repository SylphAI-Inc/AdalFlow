Generator
============
Generator is the most essential functional component in LightRAG. 
It is a user-facing orchestration component for LLM prediction.
It orchestrates the following components along with their required arguments:
- A prompt template
- Model client
- Output processors

By switching out the model client, you can call any LLM model on your prompt.

GeneratorOutput
^^^^^^^^^^^^^^^
Different from all other components, we can not alway enforce LLM to output the right format.
We in particular created an output data class to track raw string response along its parsed task response and error messages for any failured LLM predictions.

Tracing
^^^^^^^
We provide two tracing methods to help you develop and improve the Generator:
1. Trace the history change(states) on prompt during your development process. Developers typical go through a long process of prompt optimization and it is frustrating
to lose track of the prompt changes when your current change actually makes the performance much worse.

We created a `GeneratorLogger` to handle the logging and saving into json files. To further simplify developers's process,
we provides a class decorator `trace_generator` where a single line of code can be added to any of your task component which
has attributes of `Generator` type automatically.

.. code-block:: python
    from tracing import trace_generator
    from core.component import component

    @trace_generator()
    class SimpleQA(component):
        def __init__(self):
            super().__init__()
            self.generator = Generator(...)
            self.generator_2 = Generator(...)
        def call(...):

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
        ]
    }
    
2. Trace all failed LLM predictions for further improvement.


Refer `use_cases/tracing` for more details.


Training
^^^^^^^^



A Note on Tokenization#
By default, LlamaIndex uses a global tokenizer for all token counting. This defaults to cl100k from tiktoken, which is the tokenizer to match the default LLM gpt-3.5-turbo.

If you change the LLM, you may need to update this tokenizer to ensure accurate token counts, chunking, and prompting.