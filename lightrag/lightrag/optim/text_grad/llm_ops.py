# class LLMCall(Function):
#     """
#     The class to define a function that can be called and backpropagated through.
#     """

#     def __init__(self, generator: Generator):
#         super().__init__()
#         self.generator = generator

#     def forward(self, prompt: str, **kwargs) -> Generator:
#         return self.generator(prompt, **kwargs)

#     def backward(self, prompt: str, **kwargs):
#         return self.generator.backward(prompt, **kwargs)

#     def __call__(self, prompt: str, **kwargs):
#         return self.forward(prompt, **kwargs)

#     def __repr__(self):
#         return f"{self.generator}"
