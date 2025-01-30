"""Code for tutorial: https://adalflow.sylph.ai/tutorials/component.html"""

from adalflow.core import Component, Generator
from adalflow.components.model_client import OpenAIClient


class EnhanceQuery(Component):
    def __init__(self):
        super().__init__()

    def call(self, query: str) -> str:
        return query + "Please be concise and only list the top treatments."


template_doc = r"""<START_OF_SYS_PROMPT> You are a doctor <END_OF_SYS_PROMPT>
<START_OF_USER_PROMPT> {{input_str}} <END_OF_USER_PROMPT>"""


class DocQA(Component):
    def __init__(self):
        super().__init__()
        self.doc = Generator(
            template=template_doc,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-3.5-turbo"},
        )

    def call(self, query: str) -> str:
        return self.doc(prompt_kwargs={"input_str": query}).data


if __name__ == "__main__":
    from adalflow.utils import setup_env

    setup_env()
    doc = DocQA()
    states = doc.to_dict()
    # print(states)
    # print(doc.__dict__)

    doc2 = DocQA.from_dict(states)
    # print(doc2.__dict__)
    # print(doc2.to_dict())
    print(doc2)

    # to_dict and from_dict should be the same
    assert doc2.to_dict() == doc.to_dict(), "to_dict and from_dict should be the same"

    print(doc("What is the best treatment for headache?"))
    print(doc2("What is the best treatment for headache?"))

    # # list other subcomponents
    # for subcomponent in doc.named_components():
    #     print(subcomponent)

    # doc.register_parameter("demo", param=Parameter(data="demo"))

    # # list all parameters
    # for param in doc.named_parameters():
    #     print(param)

    # print(doc.to_dict())

    # from adalflow.utils.file_io import save_json

    # save_json(doc.to_dict(), "doc.json")

    # print(doc.state_dict())
    # print(doc.call("What is the best treatment for a cold?"))

    # enhance_query = EnhanceQuery()

    # seq = Sequential(enhance_query, doc)
    # query = "What is the best treatment for headache?"
    # print(seq(query))
    # print(seq)
