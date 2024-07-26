from lightrag.optim.parameter import Parameter
from lightrag.core import Component, Generator
from lightrag.components.model_client.groq_client import GroqAPIClient
from lightrag.components.model_client.openai_client import OpenAIClient
from lightrag.optim.text_grad.llm_text_loss import LLMAsTextLoss
from lightrag.optim.text_grad.textual_grad_desc import TextualGradientDescent
from lightrag.utils import setup_env

# logger = get_logger(level="DEBUG", filename="adalflow.log")

setup_env()
llama3_model = {
    "model_client": GroqAPIClient(),
    "model_kwargs": {
        "model": "llama-3.1-8b-instant",
    },
}
gpt_3_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "gpt-3.5-turbo",
        "max_tokens": 2000,
        "temperature": 0.0,
        "top_p": 0.99,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
    },
}

gpt_4o_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "gpt-4o",
    },
}


# TODO: add this to generator, we will get all parmeters and pass it to the optimizer
question_string = (
    "If it takes 1 hour to dry 25 shirts under the sun, "
    "how long will it take to dry 30 shirts under the sun? "
    "Reason step by step"
)


# TODO: test the optimization of a prompt
# this is instance optimization, it also be a single step prompt optimization
class SimpleQA(Component):
    def __init__(self, model_client, model_kwargs):
        super().__init__()
        self.model_client = model_client
        self.model_kwargs = model_kwargs

        prompt_kwargs = {
            "task_desc_str": Parameter(
                data="Answer the following question",
                requires_opt=True,
                role_desc="task description prompt to the LLM",
            ),
        }

        self.llm = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
        )

    # TODO: add this in parameter
    def call(self, question: str):
        prompt_kwargs = {
            "input_str": Parameter(
                data=question, requires_opt=False, role_desc="input to the LLM"
            )
        }
        return self.llm(prompt_kwargs)  # use forward method


# Might be extended to optimize hyperparameters
qa = SimpleQA(**gpt_4o_model)
qa.train()
print(qa.llm.training)

answer = qa.call(question_string)
print("eval answer: ", answer)

answer_param = answer
answer_param.role_desc = "The response of the LLM"
# answer_param = Parameter(
#     data=answer, requires_opt=True, role_desc="The response of the LLM"
# )
loss_fn = LLMAsTextLoss(
    prompt_kwargs={
        "eval_system_prompt": (
            f"Here's a question: {question_string}. "
            "Evaluate any given answer to this question, "
            "be smart, logical, and very critical. "
            "Just provide concise feedback."
        )
    },
    **gpt_4o_model,
)
print(f"loss_fn: {loss_fn}")
import itertools

params = list(itertools.chain(qa.parameters(), [answer_param]))
for p in qa.named_parameters():
    print(f"qa.parameters: {p}")


optimizer = TextualGradientDescent(
    params=params,  # noqa: F841
    **gpt_4o_model,
)
print(f"optimizer: {optimizer}")

l = loss_fn(prompt_kwargs={"eval_user_prompt": answer_param})  # noqa: E741
print(f"l: {l}")
l.backward()
dict_data = l.to_dict()
print(f"dict_data: {dict_data}")
# save dict_data to a file
# save_json(dict_data, "dict_data.json")
optimizer.step()  # this will update x prameter
# print(f"optimized answer: {answer_param}")
for param in params:
    print(f"optimized param: {param}")
