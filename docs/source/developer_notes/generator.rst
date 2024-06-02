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

Training
^^^^^^^^
1. Trace the error messages
2. Trace the state_dict of the model. 



A Note on Tokenization#
By default, LlamaIndex uses a global tokenizer for all token counting. This defaults to cl100k from tiktoken, which is the tokenizer to match the default LLM gpt-3.5-turbo.

If you change the LLM, you may need to update this tokenizer to ensure accurate token counts, chunking, and prompting.