from lightrag.optim.parameter import Parameter
from lightrag.components.model_client.groq_client import GroqAPIClient
from lightrag.components.model_client.openai_client import OpenAIClient
from lightrag.optim.text_grad_optimizer import LLMAsTextLoss
from lightrag.optim.text_grad.textual_grad_desc import TextualGradientDescent
from lightrag.utils import setup_env, get_logger

logger = get_logger(level="DEBUG", filename="adalflow.log")

setup_env()

# TODO: add this to generator, we will get all parmeters and pass it to the optimizer
x = Parameter(
    data="A sntence with a typo",
    role_desc="The input sentence",
    requires_opt=True,
)  # weights

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

eval_system_prompt = Parameter(
    data="Evaluate the correctness of this sentence",
    role_desc="The system prompt",
    requires_opt=True,
)
# TODO: Only generator needs parameters to optimize
loss_fn = LLMAsTextLoss(
    prompt_kwargs={
        "eval_system_prompt": eval_system_prompt,
    },
    **gpt_3_model,
)
print(f"loss_fn: {loss_fn}")

optimizer = TextualGradientDescent(params=[x, eval_system_prompt], **gpt_3_model)
print(f"optimizer: {optimizer}")

l = loss_fn(prompt_kwargs={"eval_user_prompt": x})  # noqa: E741
print(f"l: {l}")
l.backward()
logger.info(f"l: {l}")
optimizer.step()  # this will update x prameter
