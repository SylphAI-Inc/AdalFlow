# LightRAG
1000 lines of code are all you need. No sales from vendors, no lock-in to frameworks, only the best practices of RAGs. 

## What is LightRAG?
We are not a framework. We do not want you to directly install the package. We want you to carefully decide to take modules and structures from here to build your own library and applications. This is a cookbook organized uniquely for easy understanding: you can read the 1000 lines of code to see a typical RAG end-to-end without jumping between files and going through multi-level class inheritance.

**Our principles:**

- We recognize that building a RAG is a highly complex and customized process, and no framework can offer that flexibility. 
- We seek to understand, not to establish. 
- We seek to be flexible, not rigid.
- We seek to be organized and clear, not complex and confusing.
- We seek to be open, with open contributions from the community and research on use cases, maybe we will come up with a more established framework in the future. 
- We seek to learn from code; we don't read documentation that calls very high-level functions and classes. 

Note: If we can't understand your code quickly and easily, you should not add it to the library.

This is a new beginning, where all developers can come together, enjoy learning and building, have fun, and build the best RAGs in the world.

We stay neutral to all frameworks and all vendors, and we do not allow vendors to merge code without a clear, articulate community vouch for their performance, and comparison with other vendor or open-source solutions.

**Our opinions:**

We are very opinionated but we ground our opinions on RAG adapters. Here are some of our unique opinions:
- LLM is a "text-in-text-out" model, we will never use model provider's APIs that manipulate our prompts. Two examples are OpenAI's 'role', 'function call', 'json output mode'. The problem with this: our prompt cannot be directly adapted to other LLM providers and we lose transparency, adding more uncontrollable variables to our system.
- Generally, we put all `prompts` together instead of separating them into multiple strings or variables. We are able to do so using the `jinja2` templating engine, which can easily take in variables and customize or add comments to the whole prompt system.

# Structure
## Foundation
- `lightrag/`: All core data structures, the core 1000 lines of code to cover the essence of a performant RAG.
## Building blocks
- `extend/`: All modules that can be used to extend the core RAG. 
  1. Mainly including functional modules: we can extend different `Embedder`, `Retriever`, `Generator`, `Agent`, `Reranker`, etc.
  2. Tracking or monitoring modules: such as `CallbackManager`, `Phoenix` integration. When necessary, please add a `README.md` to explain why this module is necessary and how to use it.
- `tests/`: All tests for the core RAG and its extensions.

## End-to-end applications
- `use_cases/`: All use cases that can be solved using LightRAG. For instance, we can solve `Question Answering`, `Summarization`, `Information Extraction`, etc.

To add a new use case, you can add a new folder in `use_cases/` with the name `application_name` and add the following files:
- `/prompt`: a directory containing all prompts used in the application.
- `/data`: a directory containing all data used in the application, or instructions on how to download the data.
- `/main.py`: a file containing the main code to run the application.

# What is considered not part of LightRAG?
- Data processing: For instance, llamaindex has `from llama_index.core.ingestion import IngestionPipeline` which transforms the data that are either in the `Document` or `Chunk`. We do not cover this in LightRAG.
  Similarly, `from llama_index.core.postprocessor import SimilarityPostprocessor` which processes the retrieved `chunk`, sometimes with further filtering or reranking. We think `reranker` is an extension of the core RAG and should be in the `extend/` folder.
