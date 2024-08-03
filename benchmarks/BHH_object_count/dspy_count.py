import dspy
from use_cases.question_answering.bhh_object_count.data import _parse_integer_answer


turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=turbo)


class GenerateAnswer(dspy.Signature):
    # NOTE: this doc string acts as the system prompt
    """You will answer a reasoning question. Think step by step."""  # The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."""

    question = dspy.InputField(desc="The question to be answered")
    answer: str = dspy.OutputField(
        desc="The numerical answer, must be an integer, dont use english words to spell out the number"
    )  # field can only be str or list of str


class ObjectCount(dspy.Module):
    def __init__(self):
        super().__init__()

        # self.generate_answer = dspy.Predict(GenerateAnswer)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):

        pred = self.generate_answer(question=question)
        answer = _parse_integer_answer(pred.answer, only_first_line=False)
        answer = str(answer)  # dspy will assume these fields are strings not integers
        print(f"Pred: {pred}, Answer: {answer}")
        return dspy.Prediction(answer=answer)


if __name__ == "__main__":
    from lightrag.utils import setup_env

    setup_env()
    obj = ObjectCount()
    question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"

    print(obj(question))
