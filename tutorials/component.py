import re
from adalflow.core import Component, Generator
from adalflow.components.model_client import OpenAIClient
from adalflow.components.model_client import GroqAPIClient
from adalflow.utils import setup_env # make sure you have a .env file with OPENAI_API_KEY and GROQ_API_KEY

from getpass import getpass
import os

# Prompt user to enter their API keys securely
openai_api_key = getpass("Please enter your OpenAI API key: ")
# Set environment variables
os.environ['OPENAI_API_KEY'] = openai_api_key
print("API keys have been set.")

template_doc = r"""<SYS> You are a doctor </SYS> User: {{input_str}}"""

from adalflow.utils import get_logger
get_logger()

class DocQA(Component):
    def __init__(self):
        super(DocQA, self).__init__()
        self.doc = Generator(
            template=template_doc,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-3.5-turbo"},
        )

    def call(self, query: str) -> str:
        return self.doc(prompt_kwargs={"input_str": query}).data
    
doc = DocQA()
# states
states = doc.to_dict()
print(states)
print(doc.__dict__)

# restore the states
doc2 = DocQA.from_dict(states)
# print(doc2.call("What is the capital of France?"))
print(doc2.__dict__)
print(doc2.to_dict())

print(doc2.to_dict() == doc.to_dict())
doc2.to_dict()

print(doc("What is the best treatment for headache?"))
print(doc2("What is the best treatment for headache?"))

# list other subcomponents
for subcomponent in doc.named_components():
    print(subcomponent)

from adalflow.optim.parameter import Parameter
doc.register_parameter("demo", param=Parameter(data="demo"))

# list all parameters
for param in doc.named_parameters():
    print(param)

print(doc.to_dict())

from adalflow.utils.file_io import save_json
save_json(doc.to_dict(), "doc.json")

print(doc.state_dict())
print(doc.call("What is the best treatment for a cold?"))

from adalflow.core.component import FunComponent

def add_one(x):
    return x + 1

fun_component = FunComponent(add_one)
print(fun_component(1))
print(type(fun_component))
# output:
# 2
# <class 'core.component.FunComponent'>

from adalflow.core.component import fun_to_component

fun_component = fun_to_component(add_one)
print(fun_component(1))
print(type(fun_component))
# output:
# 2
# <class 'adalflow.core.component.AddOneComponent'>

# use it as a decorator
@fun_to_component
def add_one(x):
    return x + 1
print(add_one(1))
print(type(add_one))
# output:
# 2
# <class 'adalflow.core.component.AddOneComponent'>

from adalflow.core import Sequential

@fun_to_component
def enhance_query(query:str) -> str:
    return query + "Please be concise and only list the top treatments."

seq = Sequential(enhance_query, doc)
query = "What is the best treatment for headache?"
print(seq(query))
print(seq)

