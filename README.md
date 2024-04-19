# LightRAG
1000 lines of code is all you need. No sales from vendors, lock-in to frameworks, only the best practices of RAGs. 

## What is LightRAG?
We are not a framework. We do not want you to directly install the package. We want you to decide careful to take modules and structures from
here to build your own library and applications. This is a cookbook which orginized contents uniquely to easy understanding: 
such that you can read the 1000 lines of code to see a typical RAG end to end without jumping between files and going through
multi-level of class inheritance.

**Our principles:**

- We recognize that building RAG is highly complex and customized process and no framework can have that flexibility. 
- We seek to understand, not to establish. 
- We seek to be flexible, not to be rigid.
- We seek to be organized and clear, not to be complex and confusing.
- We seek to be open, with the open contribution from the community and research on the use cases, maybe we will come up a more established framework in the future. 
- We seek to learn from code, we dont read documentations which call very high level functions and classes. 

Note: If we cant understand from your code quickly and easily, you should not add it to the library.

This is a new beginning, where all developers can come together, enjoy the learning and building, having fun and to build the best RAGs in the world.

We stay neutral to all frameworks and all vendors, and we do not allow vendors to merge code without a clear articulate and a community vouch for their performance. And compare with other vendor or open-soure solutions.

**Our oppinions:**

We are very much oppionated but we ground our oppinions on RAG adapters, here are some of our unique oppinions:
- LLM is "text-in-text-out" model, we will never use model provider's APIs that will manipulate our prompts. Two examples are: OpenAI's 'role', 'function call', 'json output mode'. The problem with this: our prompt can not be directly adapted to other LLM provider and we lose transparency and adding more uncontrollable variables to our system.
- Generally, we put all `prompt` togehter instead of separating them into multiple strings or variables. We are able to do so with using `jinja2` templating engine, which can easily takes in variables and customize or adding comments in the whole prompt system.
# Structure
## Foundation
- `lightrag/`: All core data structures, the core 1000 lines of code to cover the essence of a performant RAG.
## Building blocks
- `extend/`: All modules that can be used to extend the core RAG. For instance, we can extend differnt `Embedder`, `Retriever`, `Generator`, `Agent`, `Reranker` etc.
- `tests/`: All tests for the core RAG and its extensions.
## End-to-end applications
- `use_cases/`: All use cases that can be solved using LightRAG. For instance, we can solve `Question Answering`, `Summarization`, `Information extraction`, etc.

To add a new use case, you can add a new folder in `use_cases/` and with name `application_name` and add the following files:
- `/prompt`: a directory containing all prompts used in the application.
- `/data`: a directory containing all data used in the application. or how to download the data.
- `/main.py`: a file containing the main code to run the application.

# What are considered not part of LightRAG?
- Data processing: For instance, llamaindex has ```from llama_index.core.ingestion import IngestionPipeline``` which transforms the data that are either on the `Document` or `Chunk`. We do not cover this in LightRAG.
  This is low value and highly customized to your data, and sometime you can use building blocks such as calling an LLM to transform the data.
  Similary, `from llama_index.core.postprocessor import SimilarityPostprocessor` which process the retrieved `chunk`, sometimes with further filtering or reranking. We think `reranker` is an extension of the core RAG, and should be in `extend/` folder.

