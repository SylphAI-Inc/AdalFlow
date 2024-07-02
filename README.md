# Introduction

LightRAG is the `PyTorch` library for building large language model (LLM) applications. We help developers with both building and optimizing `Retriever`-`Agent`-`Generator` (RAG) pipelines.
It is light, modular, and robust.

**PyTorch**

```python
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
```

**LightRAG**

```python

from lightrag.core import Component, Generator
from lightrag.components.model_client import GroqAPIClient
from lightrag.utils import setup_env #noqa

class SimpleQA(Component):
   def __init__(self):
      super().__init__()
      template = r"""<SYS>
      You are a helpful assistant.
      </SYS>
      User: {{input_str}}
      You:
      """
      self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192"},
            template=template,
      )

   def call(self, query):
      return self.generator({"input_str": query})

   async def acall(self, query):
      return await self.generator.acall({"input_str": query})
```

## Simplicity

Developers who are building real-world Large Language Model (LLM) applications are the real heroes.
As a library, we provide them with the fundamental building blocks with 100% clarity and simplicity.

* Two fundamental and powerful base classes: Component for the pipeline and DataClass for data interaction with LLMs.
* We end up with less than two levels of subclasses. Class Hierarchy Visualization.
* The result is a library with bare minimum abstraction, providing developers with maximum customizability.

Similar to the PyTorch module, our Component provides excellent visualization of the pipeline structure.

```
SimpleQA(
   (generator): Generator(
      model_kwargs={'model': 'llama3-8b-8192'},
      (prompt): Prompt(
         template: <SYS>
               You are a helpful assistant.
               </SYS>
               User: {{input_str}}
               You:
               , prompt_variables: ['input_str']
      )
      (model_client): GroqAPIClient()
   )
)
```

## Controllability

Our simplicity did not come from doing 'less'.
On the contrary, we have to do 'more' and go 'deeper' and 'wider' on any topic to offer developers maximum control and robustness.

* LLMs are sensitive to the prompt. We allow developers full control over their prompts without relying on API features such as tools and JSON format with components like Prompt, OutputParser, FunctionTool, and ToolManager.
* Our goal is not to optimize for integration, but to provide a robust abstraction with representative examples. See this in ModelClient and Retriever.
* All integrations, such as different API SDKs, are formed as optional packages but all within the same library. You can easily switch to any models from different providers that we officially support.

## Future of LLM Applications

On top of the easiness to use, we in particular optimize the configurability of components for researchers to build their solutions and to benchmark existing solutions.
Like how PyTorch has united both researchers and production teams, it enables smooth transition from research to production.
With researchers building on LightRAG, production engineers can easily take over the method and test and iterate on their production data.
Researchers will want their code to be adapted into more products too.
