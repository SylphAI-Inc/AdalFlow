from lightrag.optim.parameter import Parameter
from lightrag.core import Component, Generator
from lightrag.components.model_client.groq_client import GroqAPIClient
from lightrag.components.model_client.openai_client import OpenAIClient
from lightrag.utils import setup_env
from lightrag.eval.answer_match_acc import AnswerMatchAcc
from lightrag.core import DataClass, GeneratorOutput, fun_to_component
from lightrag.components.output_parsers import YamlOutputParser
from lightrag.optim.text_grad.textual_grad_desc import TextualGradientDescent
from dataclasses import dataclass, field
from textgrad.tasks import load_task
import textgrad as tg
import numpy as np
import random
import concurrent
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# logger = get_logger(level="DEBUG", filename="adalflow.log")

setup_env()
# Load the data and the evaluation function
llama3_model = {
    "model_client": GroqAPIClient(),
    "model_kwargs": {
        "model": "llama-3.1-8b-instant",
    },
}
gpt_3_model = {
    "model_client": OpenAIClient(input_type="text"),
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


def load_data():
    train_set, val_set, test_set, eval_fn = load_task(
        "BBH_object_counting", evaluation_api=None
    )
    print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()
    print(STARTING_SYSTEM_PROMPT)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class ObjectCountPredData(DataClass):
    thought: str = field(metadata={"desc": "List your step by step reasoning"})
    answer: int = field(
        metadata={"desc": "The answer to the question, only numerical values"}
    )


@fun_to_component
def parse_integer_answer(answer: str, only_first_line: bool = False):
    try:
        if only_first_line:
            answer = answer.strip().split("\n")[0]
        answer = answer.strip()
        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][
            -1
        ]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0

    return answer


# Build a pipeline like you normally would == PyTorch model
class ObjectCountTask(Component):
    def __init__(self, model_client, model_kwargs):
        super().__init__()
        template = r"""<SYS>{{system_prompt}}
        <OUTPUT_FORMAT> {{output_format_str}}</OUTPUT_FORMAT></SYS>
        <USER>{{input_str}}</USER>You:"""  # noqa: F841
        template_2 = r"""<START_OF_SYSTEM_PROMPT>{{system_prompt}}<OUTPUT_FORMAT> {{output_format_str}}</OUTPUT_FORMAT></END_OF_SYSTEM_PROMPT>{{input_str}}"""
        # data = (
        #     "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
        # )

        self.system_prompt = Parameter(
            data="You will answer a reasoning question. Think step by step.",
            role_desc="The system prompt",
            requires_opt=True,
        )
        self.output_format_str = Parameter(
            data="Respond with valid JSON object with the following schema:\n"
            + ObjectCountPredData.to_json_signature(),
            role_desc="The output format string",
            requires_opt=True,
        )
        parser = YamlOutputParser(
            data_class=ObjectCountPredData, return_data_class=True
        )  # noqa: F841
        self.llm_counter = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template_2,
            prompt_kwargs={
                "system_prompt": self.system_prompt,
                "output_format_str": self.output_format_str,
            },
            output_processors=parser,
        )
        # self.loss_fn = LLMAsTextLoss(

    def call(self, question: str) -> int:
        output: GeneratorOutput = self.llm_counter(
            prompt_kwargs={"input_str": question}
        )
        if output.data is None:
            logger.error(
                f"Error in processing the question: {question}, output: {output}"
            )
            return -1
        else:
            return output.data.answer


# Define a evaluator == PyTorch Evaluator
# class ObjectCountEvaluator(BaseEvaluator):


# TODO: improve cache for the training
# Write a trainer  == PyTorch Trainer
class ObjectCountTrainer(Component):
    def __init__(self, model_client, model_kwargs):
        super().__init__()
        set_seed(12)
        self.train_set, self.val_set, self.test_set, self.eval_fn = load_task(
            "BBH_object_counting", evaluation_api=None
        )
        # use this for test
        # self.train_set = self.train_set[:10]
        # self.val_set = self.val_set[:10]
        # self.test_set = self.test_set[:10]

        self.evaluator = AnswerMatchAcc()
        print(self.train_set.get_task_description())
        print(f"eval_fn: {self.eval_fn}")
        self.train_loader = tg.tasks.DataLoader(
            self.train_set, batch_size=4, shuffle=True
        )  # why not torch loader?

        self.task = ObjectCountTask(
            model_client=model_client, model_kwargs=model_kwargs
        )
        self.optimizer = TextualGradientDescent(
            params=self.task.llm_counter.prompt_kwargs.values(), **gpt_3_model
        )
        # self.STARTING_SYSTEM_PROMPT = self.train_set.get_task_description()
        # print(self.STARTING_SYSTEM_PROMPT)
        # self.train_loader = tg.tasks.DataLoader(
        #     self.train_set, batch_size=3, shuffle=True
        # )

    def train(self, max_epochs: int = 1):
        # set it to train mode
        self.training = True
        for epoch in range(max_epochs):
            pbar = tqdm(self.train_loader, position=0, desc=f"Epoch: {epoch}")
            for steps, (batch_x, batch_y) in enumerate(pbar):
                pbar.set_description(f"Epoch: {epoch}, Step: {steps}")
                self.optimizer.zero_grad()
                # batch_loss = []
                for x, y in zip(batch_x, batch_y):
                    response = self.task.call(
                        question=Parameter(
                            data=x,
                            role_desc="query to the language model",
                            requires_opt=False,
                        )
                    )
                    print(f"response: {response}")

    def eval_no_concurrent(self, dataset=None, max_samples: int = 100):
        if dataset is None:
            print("No dataset provided, using test set")
            dataset = self.test_set

        # set it to eval mode
        self.training = False
        x, y, y_pred = [], [], []
        tqdm_loader = tqdm(dataset)
        for _, sample in enumerate(tqdm_loader):
            y.append(sample[1])
            y_pred.append(self.task.call(question=sample[0]))
            x.append(sample[0])
            print(f"y: {y}, y_pred: {y_pred}, x: {x}")
            tqdm_loader.set_description(
                f"Accuracy: {self.evaluator.compute(y_pred, y)}"
            )

        return self.evaluator.compute(y_pred, y)[1]

    def eval(self, dataset=None, max_samples: int = 100):
        if dataset is None:
            print("No dataset provided, using test set")
            dataset = self.test_set

        # set it to eval mode
        self.training = False
        x, y, y_pred = [], [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for _, sample in enumerate(tqdm(dataset)):
                future = executor.submit(self.task.call, question=sample[0])
                futures.append((future, sample))  # store the sample with the future
                if max_samples and len(futures) >= max_samples:
                    break
            tqdm_loader = tqdm(
                concurrent.futures.as_completed(
                    [f[0] for f in futures]
                ),  # Pass only the futures to as_completed
                total=len(futures),
                position=0,
                desc="Evaluating",
            )
            for future in tqdm_loader:
                # Find the associated sample for the future
                associated_sample = next(
                    sample for fut, sample in futures if fut == future
                )
                y.append(associated_sample[1])
                y_pred.append(future.result())
                x.append(associated_sample[0])

                tqdm_loader.set_description(
                    f"Accuracy: {self.evaluator.compute(y_pred, y)}"
                )
                print(f"y: {y}, y_pred: {y_pred}, x: {x}")
        return self.evaluator.compute(y_pred, y)[1]

    def _extra_repr(self) -> str:
        s = f"train_set: {len(self.train_set)}, val_set: {len(self.val_set)}, test_set: {len(self.test_set)}, "
        s += f"eval_fn: {self.eval_fn}, "
        s += f"evaluator: {self.evaluator}"
        return s


# TODO: implement cache for generator(make it configurable)
if __name__ == "__main__":
    task = ObjectCountTask(**gpt_3_model)
    print(task)
    # print(
    #     task.llm_counter.print_prompt(
    #         input_str="How many musical instruments do I have?"
    #     )
    # )
    # print(
    #     task.call(
    #         "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"
    #     )
    # )

    trainer = ObjectCountTrainer(**gpt_3_model)
    print(trainer)
    # trainer.eval(max_samples=None)
    # trainer.eval(dataset=trainer.val_set, max_samples=None)
    # gpt-3.5-turbo test 0.69 [10 samples = 0.8], 0.72 (simple pasing, instead of json)
    # 0.73 with new parameters close to text-grad, using separate prompt: 0.81
    # single prompt without you: -> 0.82 <SYSTEM> system prompt.<SYS>0.78 <START_OF_SYSTEM_PROMPT> system prompt.<END_OF_SYSTEM_PROMPT> =>0.84 json_output = 0.68
    # yaml parser = 0.73  # json fixed 0.8 with different field description
    # text/ user role -> 0.76
    # so there is performance drop if we put the same prompt together
    # gpt-4o test 0.94

    # eval: 0.8
    trainer.train(max_epochs=1)

# llama3_model = {
#     "model_client": GroqAPIClient(),
#     "model_kwargs": {
#         "model": "llama-3.1-8b-instant",
#     },
# }

# # TODO: add this to generator, we will get all parmeters and pass it to the optimizer
# x = Parameter(
#     data="A sntence with a typo",
#     role_desc="The input sentence",
#     requires_opt=True,
# )  # weights


# gpt_3_model = {
#     "model_client": OpenAIClient(),
#     "model_kwargs": {
#         "model": "gpt-3.5-turbo",
#     },
# }

# eval_system_prompt = Parameter(
#     data="Evaluate the correctness of this sentence",
#     role_desc="The system prompt",
#     requires_opt=True,
# )
# # TODO: Only generator needs parameters to optimize
# loss_fn = LLMAsTextLoss(
#     prompt_kwargs={
#         "eval_system_prompt": eval_system_prompt,
#     },
#     **gpt_3_model,
# )
# print(f"loss_fn: {loss_fn}")

# optimizer = TextualGradientDescent(params=[x, eval_system_prompt], **gpt_3_model)
# print(f"optimizer: {optimizer}")

# l = loss_fn(prompt_kwargs={"eval_user_prompt": x})  # noqa: E741
# print(f"l: {l}")
# l.backward()
# logger.info(f"l: {l}")
# optimizer.step()  # this will update x prameter
