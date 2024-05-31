.. LightRAG documentation master file, created by
   sphinx-quickstart on Thu May  9 15:45:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
=======================
LightRAG documentation
=======================

.. .. image:: ../../images/lightrag_structure.png
..    :width: 60%

LightRAG is the "PyTorch" library for building large langage model(LLM) applications. It is super light, modular and robust like "PyTorch", and offers essential components for `Retriever`-`Agent`-`Generator` (RAG). 

You have a similar coding experience as PyTorch. Here is a side to side comparison of writing a PyTorch module and a LightRAG component:

#TODO: make it side to side comparison

**PyTorch:**

.. code-block:: python

   import torch
   import torch.nn as nn

   class Net(nn.Module):
      def __init__(self):
         super(Net, self).__init__()
         self.conv1 = nn.Conv2d(1, 32, 3, 1)
         self.conv2 = nn.Conv2d(32, 64, 3, 1)
         self.dropout1 = nn.Dropout2d(0.25)
         self.dropout2 = nn.Dropout2d(0.5)
         self.fc1 = nn.Linear(9216, 128)
         self.fc2 = nn.Linear(128, 10)

      def forward(self, x):
         x = self.conv1(x)
         x = self.conv2(x)
         x = self.dropout1(x)
         x = self.dropout2(x)
         x = self.fc1(x)
         return self.fc2(x)

   my_nn = Net()
   print(my_nn)

**LightRAG:**

.. code-block:: python

   from core.component import Component
   from core.generator import Generator
   from components.api_client import OpenAIClient

   class SimpleQA(Component):
      def __init__(self):
         super().__init__()
         self.generator = Generator(
            model_client=OpenAIClient,
            model_kwargs={'model_name': 'gpt-3.5-turbo'}
         )

      def call(self, query):
         return self.generator.call(query)

      async def acall(self, query):
         return await self.generator.acall(query)

   qa = SimpleQA()
   print(qa)


**Why LightRAG?**


1. **Clarity and Simplicity**

   We understand that developers building real-world Large Language Model (LLM) applications are the real heroes. Just like AI researchers and engineers who build models on top of PyTorch, developers require **Maximum Flexibility and Customizability**: Each developer has unique data needs to build their own models/components, experiment with In-context Learning (ICL) or model finetuning, and deploy the LLM applications to production. This means the library must provide fundamental lower-level building blocks and strive for clarity and simplicity:

   - We maintain no more than two levels of subclasses.
   - Each core abstract class is designed to be robust and flexible.
   - We use 10X less code than other libraries to achieve 10X more robustness and flexibility.


2. **Control and Transparency**

   Coming from a deep AI research background, we understand that the more control and transparency developers have over their prompts, the better. In default:

   - LightRAG simplifies what developers need to send to LLM proprietary APIs to just two messages each time: a `system message` and a `user message`. This minimizes reliance on and manipulation by API providers.
   - LightRAG provides advanced tooling for developers to build `agents`, `tools/function calls`, etc., without relying on any proprietary API provider's 'advanced' features such as `OpenAI` assistant, tools, and JSON format.

3. **Suitted for Both Researchers and Production Engineers**

   On top of the easiness to use, we in particular optimize the configurability of components for researchers to build their solutions and to benchmark existing solutions. 
   Like how PyTorch has united both researchers and production teams, it enables smooth transition from research to production. 
   With researchers building on LightRAG, production engineers can easily take over the method and test and iterate on their production data. 
   Researchers will want their code to be adapted into more products too. 
   


**LightRAG vs other LLM libraries:**


**LightRAG library structures as follows:**

#TODO: One diagram to make people understand lightrag faster

* `core` - Base abstractions, core functions, and core components like `Generator` and `Embedder` to support more advanced components.
* `components` - Components that are built on top of the core directive. Users will install relevant depencides on their own for some components.


**LightRAG documentation is divided into two parts:**

* **Developer Documentation**: This documentation explains how LightRAG is designed in more depth and is especially useful
 for developers who want to contribute to LightRAG.

* **User Documentation**: This documentation is for users who want to use LightRAG to build their applications.

We encourage all users to at least skim through the developer documentation. Different from "PyTorch" where a normal user does not have to customize a building module for neural network, 
LLM applications have much bigger scope and varies even more to different product environments, so developers customizing components on their own is much more common.

Dive deep into the design of the libraries
=======================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Community

   community/index

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer Notes

   developer_notes/index


.. toctree::
   :maxdepth: 1
   :caption: API Reference

   apis/index

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks
   .. Manually add documents for the code in benchmarks


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Resources
   
   resources/index


Use the library
=======================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Get Started

   get_started/index


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/index
