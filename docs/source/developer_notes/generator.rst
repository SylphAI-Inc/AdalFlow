class Generator
============


A Note on Tokenization#
By default, LlamaIndex uses a global tokenizer for all token counting. This defaults to cl100k from tiktoken, which is the tokenizer to match the default LLM gpt-3.5-turbo.

If you change the LLM, you may need to update this tokenizer to ensure accurate token counts, chunking, and prompting.