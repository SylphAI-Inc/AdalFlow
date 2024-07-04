![LightRAG Logo](https://raw.githubusercontent.com/SylphAI-Inc/LightRAG/main/docs/source/_static/images/LightRAG-logo-doc.jpeg)


### ⚡⚡⚡ The PyTorch Library for Large language Model (LLM) Applications ⚡⚡⚡

*LightRAG* helps developers with both building and optimizing *Retriever-Agent-Generator (RAG)* pipelines.
It is *light*, *modular*, and *robust*.



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

## Quick Install

Install LightRAG with pip:

```bash
pip install lightrag
```

Please refer to the [full installation guide](https://lightrag.sylph.ai/get_started/installation.html) for more details.



# Documentation

LightRAG full documentation available at [lightrag.sylph.ai](https://lightrag.sylph.ai/):

- [Introduction](https://lightrag.sylph.ai/)
- [Full installation guide](https://lightrag.sylph.ai/get_started/installation.html)
- [Design philosophy](https://lightrag.sylph.ai/developer_notes/lightrag_design_philosophy.html): Design based on three principles: Simplicity over complexity, Quality over quantity, and Optimizing over building.
- [Class hierarchy](https://lightrag.sylph.ai/developer_notes/class_hierarchy.html): We have no more than two levels of subclasses. The bare minimum abstraction provides developers with maximum customizability and simplicity.
- [Tutorials](https://lightrag.sylph.ai/developer_notes/index.html): Learn the `why` and `how-to` (customize and integrate) behind each core part within the `LightRAG` library.
- [API reference](https://lightrag.sylph.ai/apis/index.html)



## Contributors

[![contributors](https://contrib.rocks/image?repo=SylphAI-Inc/LightRAG&max=2000)](https://github.com/SylphAI-Inc/LightRAG/graphs/contributors)

# Citation

```bibtex
@software{Yin2024LightRAG,
  author = {Li Yin},
  title = {{LightRAG: The PyTorch Library for Large Language Model (LLM) Applications}},
  month = {7},
  year = {2024},
  doi = {10.5281/zenodo.12639531},
  url = {https://github.com/SylphAI-Inc/LightRAG}
}
```
