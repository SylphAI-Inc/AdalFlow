import logging


log = logging.getLogger(__name__)


if __name__ == "__main__":

    # from use_cases.classification.config_log import log
    # from lightrag.utils import save_json

    # train_dataset, eval_dataset, test_dataset = load_datasets()
    # # TODO: ensure each time the selected eval and test dataset and train dataset are the same
    # num_shots = 6
    # batch_size = 10
    # trainer = TrecTrainer(
    #     num_classes=6,
    #     train_dataset=train_dataset,  # use for few-shot sampling
    #     eval_dataset=eval_dataset,  # evaluting during icl
    #     test_dataset=test_dataset,  # the final testing
    #     num_shots=num_shots,
    #     batch_size=batch_size,
    # )

    # # save the most detailed trainer states
    # # When your dataset is small, this json file can be used to help you visualize datasets
    # # and to debug components
    # save_json(
    #     trainer.to_dict(),
    #     "use_cases/classification/traces/trainer_states.json",
    # )
    # log.info(f"trainer to dict: {trainer.to_dict()}")
    # # or log a str representation, mostly just the structure of the trainer
    # log.info(f"trainer: {trainer}")
    # # trainer.train_instruction(max_steps=1)
    # # trainer.train(shots=num_shots, max_steps=20, start_shots=6)
    # # trainer.eval_zero_shot()
    # trainer.eval_few_shot(shots=num_shots, runs=5)
    from textgrad.engine import get_engine
    from textgrad import Variable, TextualGradientDescent
    from textgrad.loss import TextLoss
    from dotenv import load_dotenv
    from lightrag.utils import get_logger

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
