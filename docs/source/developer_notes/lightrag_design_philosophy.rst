LightRAG Design Philosophy
====================================

.. Deep understanding of the LLM workflow
.. ---------------------------------------

LLMs are like `water`, it is all up to users to shape it into different use cases. In `PyTorch`, most likely users do not need to build their
own ``conv`` or ``linear`` module, or their own ``Adam`` optimizer. Their building blocks can meet > 90% of their user's needs on `building` and 
`optimizing` (training) their models, leaving less than 10% of users, mostly contributors and researchers to build their own ``Module``, ``Tensor``, 
``Optimizer``, etc. Libraries like `PyTorch`, `numpy`, `scipy`, `sklearn`, `pandas`, etc. are all doing the heavy lifting on the computation optimization.
However, for developers to write their own LLM task pipeline, calling apis or using local LLMs to shape the LLMs via prompt into any use case is not a hard feat.
The hard part is on `evaluating` and `optimizing` their task pipeline.

Optimizing over Building 
-----------------------------------------------------------------------

 We help users to build the task pipeline, but we want to help on optimizing even more so. 

In fact, building the task pipeline accounts for only **10%** of users' development process, the other **90%** is on optimtizing and iterating.
The most popular libraries like ``Langchain`` and ``LlamaIndex`` are mainly focusing on `building` the task pipeline, prioritizing integrations and coveraging on different type of tasks, resulting large amounts of classes, each 
with many layers of class inheritance. With the existing libraries, users get stuck on just following the examples, and it requires more time for them to figure out customization than writing their 
own code.

How to `build` the task pipeline has starting to mature: `prompt`, `retriever`, `generator`, `RAG`, `Agent` has becoming well-known concepts.
How to `optimize` the task pipeline is still a mystery to most users. And most are still doing `manual` prompt engineering without good 
`observability` (or `debugging` ) tools. And these existing `observability` tools are mostly commercialized, prioritizing the `fancy` looks without
real deep understanding of the LLM workflow.

The existing optimization process of LLM applications are full of frustrations.

Quality over Quantity
-----------------------------------------------------------------------

 The Quality of core building blocks over the Quantity of integrations.

The whole `PyTorch` library is built on a few core and base classes: ``Module``, ``Tensor``, ``Parameter``, and ``Optimizer``, 
and various ``nn`` modules for users to build a model, along with ``functionals``.
This maps to ``Component``, ``DataClass``,  ``Parameter``, and ``Optimizer`` in LightRAG, and various subcomponents 
like ``Generator``, ``Retriever``, ``Prompt``, ``Embedder``, ``ModelClient``, along with ``functionals`` to process string,
interprect tool from the string.

We recognize developers who are building real-world Large Language Model (LLM) applications are the real heroes, doing the hard
work. They need well-designed core building blocks:  **easy** to understand, **transparent** to debug, **flexible** enough to customize their own
``ModelClient``, their own ``Prompt``, their own ``Generator`` and even their own ``Optimizer``, ``Trainer``. The need to build their own component is even more so than using `PyTorch.`
LightRAG aggressively focus on the quality and clarity of the core building blocks over the quantity of integrations.



Practicality over Showmanship
-----------------------------------------------------------------------
We put these three hard rules while designing LightRAG:

- Every layer of abstraction needs to be adjusted and overall we do not allow more than 3 layers of abstraction.
- We minimize the lines of code instead of maximizing the lines of code.
- Go `deep` and `wide` in order to `simplify`.  The clarity we achieve is not the result of being easy, but the result of being deep.





[Optional] Side story: How `LightRAG` is born
----------------------------------------------