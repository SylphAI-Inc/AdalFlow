from dspy import Signature, InputField, OutputField, Predict
import dspy
import utils.setup_env
from use_cases.classification.data import (
    extract_class_label,
)

turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=250)
dspy.settings.configure(lm=turbo)


class TrecClassifier(Signature):
    """You are a classifier. Given a Question, you need to classify it into one of the following classes:
    Format: class_index. class_name, class_description
    0. ABBR, Abbreviation
    1. ENTY, Entity
    2. DESC, Description and abstract concept
    3. HUM, Human being
    4. LOC, Location
    5. NUM, Numeric value"""

    question = InputField(
        name="question", type=str, description="The question to classify"
    )
    class_name = OutputField(
        name="class_name",
        type=str,
        description="class_name",
    )
    class_index = OutputField(
        name="class_index",
        type=int,
        description="class_index in range[0, 5]",
    )


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(signature=TrecClassifier)

    def forward(self, question):
        pred = self.prog(question=question)

        return pred


if __name__ == "__main__":
    # test one example
    query = "How did serfdom develop in and then leave Russia ?"
    trec_classifier = CoT()
    print(trec_classifier)
    response = trec_classifier(query)
    print(response)
