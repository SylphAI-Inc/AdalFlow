Introduction to LLM applications
============
All language models are doing is text completion, which means you pass a input text(called "prompt") to the model, and the model will generate the rest of the text for you. The model is trained to predict the next word given the previous words, so it can generate text that is coherent and relevant to the prompt.
However, most API providers such as OpenAI, Anthropic provide the model in Chat Completions API, which take a list of messages instead of a `single text` as input. Why is that? We will have our assumptions in the next section.

Here is one example from OpenAI:

.. code-block:: python
   :linenos:

   from openai import OpenAI
   client = OpenAI()
   
   response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
    )

You can notice the first is the "system" message, and with interleave of "user" and "assistant" messages after that.
Different API providers may have different input format, e.g. Anthropic's system is a field itself and requires a string.

.. code-block:: python
   :linenos:

   import anthropic

    anthropic.Anthropic().messages.create(
        model="claude-3-opus-20240229",
        system="You are a helpful assistant.",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, world"}
        ]
    )

Why do we need chat completion instead of the previous text completion?
---------------------
There are generally three roles("system", "user", "assistant") in chat, and potentially more especially in multi-agent. 
Behind the scene, when the API provider receives the messages, they will compose them into a single text using special tokens to denote different messages.
If we follow `Llama3 special tokens <https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/>`_, the above anthropic example will be converted into text:

.. code-block::
   :linenos:

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant. <|eot_id>
    <|start_header_id|>user<|end_header_id|>
    Hello, world <|eot_id>
    <|start_header_id>assistant<|end_header_id>

And the model will generate the text response with `<|end_of_text|>` token to denote the end of the text.

The special tokens vary from different language models, and with propetriary models, most of them dont disclose the special tokens. Why?

1. They keep more secrets about their model.

2. Without knowing the special tokens, developers will be more dependant on their API, especially for advanced features like `Tools`, `Special output format`.

But, you don't have to use their advanced features, our library in default will support those advanced features, ensuring developers to have `maximum control` and be almost model-agnostic.

**Read on to see how.**

Resources
---------------------
[1] `OpenAI text generation <https://platform.openai.com/docs/guides/text-generation>`_

[2] `Anthropic text generation <https://docs.anthropic.com/en/docs/system-prompts>`_

[3] `Meta Llama3 <https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/>`_