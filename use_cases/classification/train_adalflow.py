from lightrag.optim.parameter import Parameter
from lightrag.components.model_client.groq_client import GroqAPIClient
from lightrag.components.model_client.openai_client import OpenAIClient
from lightrag.optim.text_grad_optimizer import LLMAsTextLoss
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
    },
}
# TODO: Only generator needs parameters to optimize
loss_fn = LLMAsTextLoss(
    prompt_kwargs={
        "eval_system_prompt": Parameter(
            data="Evaluate the correctness of this sentence",
            role_desc="The system prompt",
            requires_opt=True,
        )
    },
    **gpt_3_model,
)
print(f"loss_fn: {loss_fn}")

l = loss_fn(prompt_kwargs={"eval_user_prompt": x})  # noqa: E741
print(f"l: {l}")
l.backward()
logger.info(f"l: {l}")
