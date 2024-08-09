import logging


log = logging.getLogger(__name__)

from dotenv import load_dotenv
from adalflow.utils import save_json

# get_logger(level="DEBUG", filename="lib_text_grad.log")

load_dotenv()


def test_text_grad():
    from textgrad.engine import get_engine
    from textgrad import Variable, TextualGradientDescent
    from textgrad.loss import TextLoss
    from dotenv import load_dotenv
    from adalflow.utils import get_logger

    get_logger(level="DEBUG", filename="lib_text_grad.log")

    load_dotenv()

    x = Variable(
        "A sntence with a typo",
        role_description="The input sentence",
        requires_grad=True,
    )  # weights
    print(x.gradients)
    engine = get_engine("gpt-3.5-turbo")
    output = engine.generate("Hello how are you?")

    print(engine)
    print(output)

    # Call it Eval Feedback, no gradient, a judge? takes y and y_hat (no y_hat) so no normal loss, but text feedback on the response.
    system_prompt = Variable(
        "Evaluate the correctness of this sentence",
        role_description="The system prompt",
    )  # this is llm
    # EvalFeedback
    loss = TextLoss(
        system_prompt, engine=engine
    )  # generate messages [{'role': 'system', 'content': 'Evaluate the correctness of this sentence'}, {'role': 'user', 'content': 'A sntence with a typo'}]
    print(loss)
    optimizer = TextualGradientDescent(
        parameters=[x], engine=engine
    )  # TODO: pass system prompt instead of x?
    print(optimizer)

    # putting together
    # loss takes x, isnt thi
    l = loss(x)  # noqa: E741
    print(f"loss: {l}")
    # computes the gradients for the variable x
    """
    v: The sentence you provided does indeed contain a typo.
    The word "sntence" should be corrected to "sentence."
    v.gradients: set()
    v: A sntence with a typo (x)
    v.gradients: {Variable(value=Since the language model correctly identified a typo in the sentence provided, the feedback for the variable "<VARIABLE> A sntence with a typo </VARIABLE>" would be to ensure that the text is free of any spelling errors before presenting it. One way to improve the variable is to run a spell check or proofread the text to catch any typos or spelling mistakes before using it in a context where accuracy is crucial. By ensuring that the text is error-free, the overall quality and credibility of the content will be enhanced, leading to better performance according to the <OBJECTIVE_FUNCTION>., role=feedback to The input sentence, grads=)}
    v: Evaluate the correctness of this sentence (prompt variable)
    v.gradients: {Variable(value=The system prompt could be improved by providing a more specific and detailed instruction to the language model. Instead of a general directive like "Evaluate the correctness of this sentence," you could consider providing more context or guidance to the model. For example, you could ask the model to specifically identify and correct any spelling errors, grammatical mistakes, or punctuation issues in the given sentence. This way, the model would have a clearer understanding of the task at hand and could provide more targeted feedback. Additionally, you could include examples of common errors that the model should look out for, which would help guide its evaluation process and improve the quality of the feedback provided., role=feedback to The system prompt, grads=)}
    """
    l.backward(engine)
    log.info(f"l: {l}")
    dict_data = l.to_dict()
    save_json(dict_data, "text_grad.json")
    # print(f"loss: {l}")
    # optimizer.step()
    # print(x)
    # print(x.gradients)

    """
    {feedback_str}
    loss: loss: The sentence you provided does indeed contain a typo. The word "sntence" should be corrected to "sentence."

    gradient: (feedback to The input sentence)
    {Variable(value=Since the language model correctly identified a typo in the sentence provided, the feedback for the variable "<VARIABLE> A sntence with a typo </VARIABLE>" would be to ensure that the text is free of any spelling errors before presenting it. One way to improve the variable is to run a spell check or proofread the text to catch any typos or spelling mistakes before using it in a context where accuracy is crucial. By ensuring that the text is error-free, the overall quality and credibility of the content will be enhanced, leading to better performance according to the <OBJECTIVE_FUNCTION>., role=feedback to The input sentence, grads=)}

    """


# ln -s /Users/liyin/Library/Caches/textgrad/ textgrad


if __name__ == "__main__":

    test_text_grad()
