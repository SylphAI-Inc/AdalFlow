Component
============
:ref:`Component<core-component>` is to LLM task pipelines is what ``nn.Module`` is to PyTorch models.
Task pipelines are composed of components, and class ``Component`` is the base class for it all.
It shares the same
the base class for all LightRAG components. It is similar to PyTorch's `nn.Module` class.
We name it differently to avoid confusion and also for better compatibility with `PyTorch`.
You write the code similar to how you write a PyTorch model.

PyTorch model code:

.. code-block:: python

    import torch.nn as nn

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.fc(x)

LightRAG component code:

.. code-block:: python

    class MyComponent(Component):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
        
        def call(self, x):
            return self.fc(x)

The printout of the model vs the component:





Here we have three major differences: 

1. `nn.Moudle` requires you to pass `Tensor` as input and even all parameters are `Tensor` as well.
In LLM applications, your input can be any data type. 

2. Instead of having a `forward` method, the main method for a component is `call` and `acall` (for asynchronous call).
In default, `__call__` method calls `call` so it is synchronous.

3. Dynamic.... 


We have the same `Sequential` class to PyTorch's `nn.Sequential` class. This is especially useful to chain together data tranformers.
