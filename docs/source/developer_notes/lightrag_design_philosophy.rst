LightRAG Design Philosophy
====================================

Deep understanding of the LLM workflow
---------------------------------------

LLMs are like `water`, it is all up to users to shape it into different use cases. In `PyTorch`, most likely users do not need to build their
own ``conv`` or ``linear`` module, or their own ``Adam`` optimizer. Their building blocks can meet > 90% of their user's needs on `building` and 
`optimizing` (training) their models, leaving less than 10% of users, mostly contributors and researchers to build their own ``Module``, ``Tensor``, 
``Optimizer``, etc. Libraries like `PyTorch`, `numpy`, `scipy`, `sklearn`, `pandas`, etc. are all doing the heavy lifting on the computation optimization.
However, for developers to write their own LLM task pipeline, calling apis or using local LLMs to shape the LLMs via prompt into any use case is not a hard feat.
The hard part is on `evaluating` and `optimizing` their task pipeline.

In fact, building the task pipeline accounts for only **10%** of users' development process, the other **90%** is on optimtizing and iterating.
The most popular libraries like ``Langchain`` and ``LlamaIndex`` are mainly focusing on `building` the task pipeline, prioritizing integrations and coveraging on different type of tasks, resulting large amounts of classes, each 
with many layers of class inheritance. With the existing libraries, users get stuck on just following the examples, and it requires more time for them to figure out customization than writing their 
own code.

How to `build` the task pipeline has starting to mature: `prompt`, `retriever`, `generator`, `RAG`, `Agent` has becoming well-known concepts.
How to `optimize` the task pipeline is still a mystery to most users. And most are still doing `manual` prompt engineering without good 
`observability` (or `debugging` ) tools. And these existing `observability` tools are mostly commercialized, prioritizing the `fancy` looks without
real deep understanding of the LLM workflow.

The existing optimization process of LLM applications are full of frustrations.

The Quality of core building blocks over the Quantity of integrations
-----------------------------------------------------------------------

.. However, for LLM applations, no library currently can provide 


.. Instead, they can use the built-in modules and optimizers. Similarly, in `LightRAG`, we provide a set of built-in modules and optimizers, and users can use them to build their own LLMs.
.. We aggressively focus on problem solving.

.. Building the task pipeline accounts for only **10%** of the development process, the other **90%** is on optimtizing and iterating,
.. via manual or auto prompting, along with hyperparmeter tuning and component optimization.


[Optional] Side story: How `LightRAG` is born
----------------------------------------------