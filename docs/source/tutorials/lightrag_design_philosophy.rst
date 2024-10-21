.. _lightrag_design_philosophy:

Design Philosophy
====================================

Right from the begining, `AdalFlow` follows three fundamental principles.


Principle 1:  Simplicity over Complexity
-----------------------------------------------------------------------
 We put these three hard rules while designing AdalFlow:

- Every layer of abstraction needs to be adjusted and overall we do not allow more than 3 layers of abstraction.
- We minimize the lines of code instead of maximizing the lines of code.
- Go *deep* and *wide* in order to *simplify*.  The clarity we achieve is not the result of being easy.



Principle 2: Quality over Quantity
-----------------------------------------------------------------------

 The Quality of core building blocks over the Quantity of integrations.

We aim to provide developers with well-designed core building blocks that are  *easy* to understand, *transparent* to debug, and *flexible* enough to customize.
This goes for the prompt, the model client, the retriever, the optimizer, and the trainer.



Principle 3: Optimizing over Building
-----------------------------------------------------------------------

 We help users build the task pipeline, but we want to help with optimizing even more so.


We design our building blocks with `optimization` in mind.
This means we go beyond just providing developers with transparency and control; we also offer excellent `logging`, `observability`, `configurability`, `optimizers`, and `trainers` to ease the existing frustrations of optimizing the task pipeline.


Our understanding of LLM workflow
-----------------------------------------------------------------------

The above principles are distilled from our experiences and continuous learning about the LLM workflow.



**Developers are the ultimate heroes**

LLMs are like `water`, they can be shaped into anything, from GenAI applications such as `chatbot`, `translation`, `summarization`, `code generation`, `autonomous agent` to classical NLP tasks like `text classification`, and `named entity recognition`.
They interact with the world beyond the model's internal knowledge via `retriever`, `memory`, and `tools` (`function calls`).
Each use case is unique in its data, its business logic, and its unique user experience.


Building LLM applications is a combination of software engineering and modeling (in-context learning).
Libraries like `PyTorch` mainly provide basic building blocks and do the heavy lifting on computation optimization.
If 10% of all `PyTorch` users need to customize a layer or an optimizer, the chance of customizing will only be higher for LLM applications.
Any library aiming to provide out-of-box solutions is destined to fail as it is up to the developers to address each unique challenge.



**Manual prompt engineering vs Auto-prompt optimization**

Developers rely on prompting to shape the LLMs into their use cases via In-context learning (ICL).
However, LLM prompting is highly sensitive: the accuracy gap between top-performing and lower-performing prompts can be as high as 40%.
It is also a brittle process that breaks the moment your model changes.
Because of this, developers end up spending **10%** of their time building the task pipeline itself, but the other **90%** in optimizing and iterating the prompt.
The process of closing the accuracy gap between the demo to the production is full of frustrations.
There is no doubt that the future of LLM applications is in auto-prompt optimization, not manual prompt engineering.
However, researchers are still trying to understand prompt engineering, the process of automating it is even more in its infancy state.

**Know where the heavy lifting is**

The heavy lifting of an LLM library is not to provide developers out-of-box prompts, not on intergrations of different API providers or data bases, it is on:

- Core base classes and abstractions to help developers on "boring" things like seralization, deserialization, standarizing interfaces, data processing.
- Building blocks to help LLMs interact with the world.
- `Evaluating` and `optimizing` the task pipeline.

All while giving full control of the prompt and the task pipeline to the developers.





.. raw::

    [Optional] Side story: How `AdalFlow` is born
.. ----------------------------------------------

.. The whole `PyTorch` library is built on a few core and base classes: ``Module``, ``Tensor``, ``Parameter``, and ``Optimizer``,
.. and various ``nn`` modules for users to build a model, along with ``functionals``.
.. This maps to ``Component``, ``DataClass``,  ``Parameter``, and ``Optimizer`` in LightRAG, and various subcomponents
.. like ``Generator``, ``Retriever``, ``Prompt``, ``Embedder``, ``ModelClient``, along with ``functionals`` to process string,
.. interprect tool from the string.

.. We recognize developers who are building real-world Large Language Model (LLM) applications are the real heroes, doing the hard
.. work. They need well-designed core building blocks:  **easy** to understand, **transparent** to debug, **flexible** enough to customize their own
.. ``ModelClient``, their own ``Prompt``, their own ``Generator`` and even their own ``Optimizer``, ``Trainer``. The need to build their own component is even more so than using `PyTorch.`
.. LightRAG aggressively focus on the quality and clarity of the core building blocks over the quantity of integrations.

.. the current state of the art in auto-prompt optimization is still in its infancy.
.. Though Auto-prompt optimization is the future, now we are still in the process of understanding more on prompt engineering itself and but it is a good starting point for auto-prompt optimization.

.. The future is at the optimizing.
.. Using LLMs via apis or local LLMs is easy, so where is the value of having a library like `LightRAG`?

.. In `PyTorch`, most likely users do not need to build their own ``conv`` or ``linear`` module, or their own ``Adam`` optimizer.
.. The existing building blocks can meet > 90% users' needs, leaving less than 10% of users, mostly contributors and researchers to build their own `Module`, `Tensor`,
.. `Optimizer`, etc. Excellent libraries like `PyTorch`, `numpy`, `scipy`, `sklearn`, `pandas` are all doing the heavy lifting on the computation optimization.


.. Using LLMs via apis or local LLMs is easy, so where is the heavy lifting in the LLM applications?
