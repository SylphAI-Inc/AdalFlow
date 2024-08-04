from lightrag.optim.parameter import Parameter, ParameterType
from lightrag.core import Component, Generator
from lightrag.core.generator import BackwardEngine
from lightrag.components.model_client.groq_client import GroqAPIClient
from lightrag.components.model_client.openai_client import OpenAIClient
from lightrag.utils import setup_env
from lightrag.eval.answer_match_acc import AnswerMatchAcc
from lightrag.eval.base import EvaluationResult

from lightrag.core import DataClass, fun_to_component
from lightrag.components.output_parsers import YamlOutputParser
from lightrag.optim.text_grad.textual_grad_desc import TextualGradientDescent
from lightrag.optim.text_grad.text_loss_with_eval_fn import EvalFnToTextLoss
from lightrag.optim.text_grad.ops import sum
from lightrag.optim._llm_optimizer import LLMOptimizer
from lightrag.datasets.big_bench_hard import BigBenchHard
from lightrag.utils import save_json
from dataclasses import dataclass, field
from textgrad.tasks import load_task
import numpy as np
from typing import Dict, Any, List, Tuple, Callable
import random
import concurrent
from tqdm import tqdm
import logging

from torch.utils.data import Subset
from lightrag.utils.data import DataLoader


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
        "temperature": 0.9,
        "top_p": 0.99,
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
# TODO: auto saving the prompt and performance.


# 1 Task: with structured output
# 2. use task pipeline instead of a single generator
# 3. train for both output format and the system prompt
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
        # 1. set up system prompt, and define the parameters for optimization.
        # NOTE: use self. will double the parameters, so we dont need that as we want the parameter to be part of the generator
        system_prompt = Parameter(
            alias="task_instruction",
            data="You will answer a reasoning question. Think step by step.",
            role_desc="To give task instruction to the language model in the system prompt",
            requires_opt=True,
            param_type=ParameterType.PROMPT,
        )
        instruction = "Do not change the fields in the JSON object. Only improve on the field descriptions."
        output_format_str = Parameter(
            alias="output_format",
            data="Respond with valid JSON object with the following schema:\n"
            + ObjectCountPredData.to_json_signature(),
            role_desc="To specify the LLM output format",
            instruction_to_optimizer=instruction,
            instruction_to_backward_engine=instruction,
            param_type=ParameterType.PROMPT,
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
                "system_prompt": system_prompt,
                "output_format_str": output_format_str,
            },
            output_processors=parser,
        )
        # TODO: make this data map function more robust (this is the final answer and the input to eval_fn)
        self.llm_counter.set_data_map_func(lambda x: x.data.answer)
        logger.info(f"llm_counter set_data_map_func, {self.llm_counter.data_map_func}")

    # TODO: the error will be a context
    def call(self, question: str) -> Any:  # Union[Parameter, int]:
        output = self.llm_counter(
            prompt_kwargs={"input_str": question}
        )  # already support both training (forward + call)

        if not self.training:  # eval

            if output.data is None:
                logger.error(
                    f"Error in processing the question: {question}, output: {output}"
                )
                output = -1
            else:
                output = output.data.answer
        return output


class ObjectCountTaskOriginal(Component):
    def __init__(self, model_client, model_kwargs):
        super().__init__()
        template = r"""<SYS>{{system_prompt}}
        <OUTPUT_FORMAT> {{output_format_str}}</OUTPUT_FORMAT></SYS>
        <USER>{{input_str}}</USER>You:"""  # noqa: F841
        template_2 = r"""<START_OF_SYSTEM_PROMPT>{{system_prompt}}<OUTPUT_FORMAT> {{output_format_str}}</OUTPUT_FORMAT></END_OF_SYSTEM_PROMPT>{{input_str}}"""
        # data = (
        #     "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
        # )
        # 1. set up system prompt, and define the parameters for optimization.
        # NOTE: use self. will double the parameters, so we dont need that as we want the parameter to be part of the generator
        system_prompt = Parameter(
            alias="task_instruction",
            # data="You will answer a reasoning question. Clearly list each intermediate step before giving the final numerical answer. Double-check each step for accuracy. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
            data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
            role_desc="To give task instruction to the language model in the system prompt",
            requires_opt=True,
            param_type=ParameterType.NONE,
        )
        self.llm_counter = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template_2,
            prompt_kwargs={
                "system_prompt": system_prompt,
            },
            output_processors=parse_integer_answer,
            use_cache=True,
        )
        # TODO: make this data map function more robust (this is the final answer and the input to eval_fn)
        # self.llm_counter.set_data_map_func(lambda x: x.data.answer)
        logger.info(f"llm_counter set_data_map_func, {self.llm_counter.data_map_func}")

    # TODO: the error will be a context
    def call(self, question: str) -> Any:  # Union[Parameter, int]:
        output = self.llm_counter(
            prompt_kwargs={"input_str": question}
        )  # already support both training (forward + call)

        if not self.training:  # eval

            if output.data is None:
                logger.error(
                    f"Error in processing the question: {question}, output: {output}"
                )
                output = -1
            else:
                output = int(output.data)
        return output


# Define a evaluator == PyTorch Evaluator
# class ObjectCountEvaluator(BaseEvaluator):
from lightrag.optim.trainer.adal import AdalComponent
from lightrag.optim.trainer.trainer import Trainer
from lightrag.datasets.big_bench_hard import ObjectCountData


class TGDWithEvalFnLoss(AdalComponent):
    def __init__(
        self,
        task_model_config: Dict,  # for task pipeline
        backward_engine_model_config: Dict,  # for computing gradients
        optimizer_model_config: Dict,  # for proposal
    ):
        super().__init__()

        self.task_model_config = task_model_config
        self.backward_engine_model_config = backward_engine_model_config
        self.optimizer_model_config = optimizer_model_config

        self.backward_engine = BackwardEngine(
            **backward_engine_model_config, use_cache=True
        )
        self.task = ObjectCountTaskOriginal(**task_model_config)
        self.evaluator = AnswerMatchAcc(type="exact_match")
        self.configure_backward_engine()

    def handle_one_train_sample(self, sample: ObjectCountData) -> Tuple[Callable, Dict]:
        return self.task.call, {"question": sample.x}

    def handle_one_loss_sample(
        self, sample: ObjectCountData, y_pred: Any
    ) -> Tuple[Callable, Dict]:
        return self.loss_fn, {
            "kwargs": {
                "y": y_pred,
                "y_gt": Parameter(
                    data=sample.y,
                    role_desc="The ground truth(reference correct answer)",
                    alias="y_gt",
                    requires_opt=False,
                ),
            }
        }

    def evaluate_one_sample(self, sample: ObjectCountData, y_pred: Any) -> Any:
        return self.evaluator.compute_single_item(y_pred, sample.y)

    def evaluate_samples(
        self, samples: List[ObjectCountData], y_preds: List
    ) -> EvaluationResult:
        r"""Support both batch and list of samples"""
        y_gts = [sample.y for sample in samples]
        return self.evaluator.compute(y_preds, y_gts)

    def train_step(self, batch, batch_idx, num_workers: int = 2) -> List:
        self.task.train()
        y_preds = super().pred_step(batch, batch_idx, num_workers)
        for i, y_pred in enumerate(y_preds):
            y_pred.alias += f"y_pred_{i}"
        return y_preds

    def configure_optimizers(self):
        return TextualGradientDescent(
            params=list(
                self.task.parameters()
            ),  # NOTE: for now it has to be a list not a generator
            **self.optimizer_model_config,
        )

    def configure_backward_engine(self):
        self.backward_engine = BackwardEngine(**self.backward_engine_model_config)
        # add backward engine to the generator of the task
        self.task.llm_counter.set_backward_engine(self.backward_engine)

    def configure_loss_fn(self):
        # share the backward engine with the generator
        if self.backward_engine is None:
            self.configure_backwar_engine()
        self.loss_fn = EvalFnToTextLoss(
            eval_fn=self.evaluator.compute_single_item,
            eval_fn_desc="ObjectCountingEvalFn, Output accuracy score: 1 for correct, 0 for incorrect",
            backward_engine=self.backward_engine,
        )


def train_object_count_text_grad_v1(
    batch_size=6, max_steps=1, max_samples=2, num_workers=2, strategy="random"
):

    trainer = Trainer(
        optimizer_type="text-grad",
        strategy=strategy,
        max_steps=max_steps,
        num_workers=num_workers,
        adaltask=TGDWithEvalFnLoss(gpt_3_model, gpt_4o_model, gpt_4o_model),
        ckpt_path="object_count_text_grad_random",
    )
    # train_dataset, val_dataset, test_dataset, eval_fn = load_task(
    #     "BBH_object_counting", evaluation_api=None
    # )
    root = "cache_datasets"
    train_dataset = BigBenchHard("BBH_object_counting", split="train", root=root)
    val_dataset = BigBenchHard("BBH_object_counting", split="val", root=root)
    test_dataset = BigBenchHard("BBH_object_counting", split="test", root=root)

    def subset_dataset(dataset, num_samples):
        num_samples = min(num_samples, len(dataset))
        random_subset_indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, random_subset_indices)

    train_dataset = subset_dataset(train_dataset, max_samples)
    val_dataset = subset_dataset(val_dataset, max_samples)
    test_dataset = subset_dataset(test_dataset, max_samples)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    trainer.fit(
        train_loader=train_dataloader,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )


class ORPOObjectCount(AdalComponent):
    def __init__(
        self,
        task_model_config: Dict,
        optimizer_model_config: Dict,
    ):
        super().__init__()

        self.task_model_config = task_model_config
        self.optimizer_model_config = optimizer_model_config

        self.task = ObjectCountTaskOriginal(**task_model_config)
        self.evaluator = AnswerMatchAcc(type="exact_match")

    def configure_optimizers(self):
        return LLMOptimizer(  # only support one parameter for now
            params=[self.task.llm_counter.system_prompt],
            **self.optimizer_model_config,
        )


# TODO: improve cache for the training
# Write a trainer  == PyTorch Trainer
class ObjectCountTrainer(Component):
    __doc__ = r"""Text-grad trainer will require:
    - Task pipeline that defines parameters
    - Optimizer and its model parameters
    - Backward engine(to compute gradients) and its model parameters
    """

    def __init__(
        self,
        task_model_config: Dict,
        backward_engine_model_config: Dict,
        tgd_model_config: Dict,
        batch_size: int = 6,
    ):
        super().__init__()
        set_seed(12)
        self.train_set, self.val_set, self.test_set, self.eval_fn = load_task(
            "BBH_object_counting", evaluation_api=None
        )

        self.evaluator = AnswerMatchAcc(type="exact_match")
        self.training_batch_size = batch_size
        print(self.train_set.get_task_description())
        print(f"eval_fn: {self.eval_fn}")
        # self.train_loader = tg.tasks.DataLoader(
        #     self.train_set, batch_size=self.training_batch_size, shuffle=True
        # )  # why not torch loader?
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.training_batch_size, shuffle=True
        )

        # self.task = ObjectCountTask(**task_model_config)
        self.task = ObjectCountTaskOriginal(**task_model_config)
        # 2. backward engine will be used by all operators
        backward_engine = BackwardEngine(**backward_engine_model_config)
        self.target_params = set(self.task.parameters())

        for param in self.target_params:
            print(f"param: {param.alias}")

        # 3. optimizer will be used to optimize the parameters
        self.orpo_optimizer = LLMOptimizer(
            params=self.target_params,
            **tgd_model_config,
        )
        self.optimizer = TextualGradientDescent(
            params=self.target_params,
            **tgd_model_config,
            # constraints=[
            #     "Do not stray too far from the original value.",
            #     "Do not be too specific to the training data to adapt to new data.",
            #     "keep the initial instruction's purpose.",
            # ],
        )

        self.task.llm_counter.set_backward_engine(backward_engine)

        # track the tokens
        self.task_total_prompt_tokens = 0
        self.task_total_completion_tokens = 0
        self.backward_engine_total_prompt_tokens = 0
        self.backward_engine_total_completion_tokens = 0
        self.optimizer_total_prompt_tokens = 0
        self.optimizer_total_completion_tokens = 0

        # 4. loss function will be used to compute the loss

        # TODO: set backward_engine should be recursive
        # pred_answer: object, gt_answer: object for compute_single_item
        self.loss_fn = EvalFnToTextLoss(
            eval_fn=self.evaluator.compute_single_item,
            eval_fn_desc="ObjectCountingEvalFn, Output accuracy score: 1 for correct, 0 for incorrect",  # NOTE: important to explain to optimizer what the metric mean.
            backward_engine=backward_engine,
        )

    def _get_param_values(self):
        return {p.alias: p.data for p in self.task.parameters() if p.requires_opt}

    def fit_v1(
        self,
        max_steps: int = 3,
        max_samples=20,
        results: Dict = None,
    ):
        # TODO: save a best prompt or top 2
        self.task.train()
        self.optimizer.zero_grad()
        logger.info(f"Training started: {self.task.training}")
        max_steps = max_steps
        max_samples = max_samples
        task_name = self.task.__class__.__name__
        save_result_file_path = f"results_adalflow_task_v1_{task_name}_max_steps_{max_steps}_max_samples_{max_samples}.json"
        # TODO: compute the epoch based on the number of samples
        for steps, (batch_x, batch_y) in enumerate(
            (pbar := tqdm(self.train_loader, position=0))
        ):
            pbar.set_description(f"Training Step: {steps}")
            self.task.train()

            losses: List[Parameter] = []
            for i, (x, y) in enumerate(zip(batch_x, batch_y)):
                # compute loss on one data point
                logger.info(f"x: {x}, y: {y}")
                response = self.task.call(
                    question=Parameter(
                        data=x,
                        role_desc="query to the language model",
                        requires_opt=False,
                        alias=f"x_{i}",
                    )
                )
                logger.info(f"response: {response}")
                response.alias += f"_{i}"
                # TODO: when it is train, need to pass the data to be something used for eval.
                loss = self.loss_fn(
                    kwargs={
                        "y": response,
                        "y_gt": Parameter(
                            data=y,
                            role_desc="The ground truth",
                            requires_opt=False,
                            alias=f"y_{i}",
                        ),
                    }
                )
                loss.alias += f"_step_{steps}_batch_{i}"
                print(f"y_gt: {y})")
                losses.append(loss)
                # loss.draw_graph(filepath="loss1")

            total_loss = sum(losses)
            print("loss backward...")
            total_loss.backward()
            print("optimizer propose...")
            self.optimizer.propose()
            prompts = self._get_param_values()
            print(f"new prompt: {prompts}")
            # total_loss.draw_graph(filepath=f"total_loss_step_{steps}")
            print("Start evaluate")

            # save_json(total_loss.to_dict(), "total_loss_adalflow.json")

            eval_acc, eval_acc_list = self.evaluate_dataset(
                dataset_type="val", max_samples=max_samples
            )
            print(f"val_acc: {eval_acc}, last acc: {results['val_acc'][-1]}")
            if eval_acc > results["val_acc"][-1]:
                print("optimizer step")
                self.optimizer.step()
                results["val_acc"].append(eval_acc)

            else:
                self.optimizer.revert()
                print("optimizer revert")
                results["val_acc"].append(results["val_acc"][-1])
            final_prompts = self._get_param_values()
            results["prompts"].append(final_prompts)

            test_acc, test_acc_list = self.evaluate_dataset(
                dataset_type="test", max_samples=max_samples
            )
            results["test_acc"].append(test_acc)
            print(f"test_acc: {test_acc}")

            save_json(results, save_result_file_path)
            if steps >= max_steps:
                break

    def fit_v2(
        self,
        max_steps: int = 3,
        max_samples=20,
        results: Dict = None,
    ):
        # TODO: save a best prompt or top 2
        self.task.train()
        self.optimizer.zero_grad()
        logger.info(f"Training started: {self.task.training}")
        max_steps = max_steps
        max_samples = max_samples
        task_name = self.task.__class__.__name__
        num_proposals = 4
        save_result_file_path = f"results_adalflow_v2_task_{task_name}_max_steps_{max_steps}_max_samples_{max_samples}.json"
        # TODO: compute the epoch based on the number of samples
        errors_losses: List[Parameter] = []
        correct_losses: List[Parameter] = []

        for steps, (batch_x, batch_y) in enumerate(
            (pbar := tqdm(self.train_loader, position=0))
        ):
            pbar.set_description(f"Training Step: {steps}")
            if steps >= max_steps:
                print(f"steps: {steps} >= max_steps: {max_steps}")
                break
            self.task.train()

            losses: List[Parameter] = []
            y_preds = self.train_batch_worker(
                batch_x
            )  # generator should always guarentee data even if it gives error
            # compute loss each data point
            for i, (x, y, y_pred) in enumerate(zip(batch_x, batch_y, y_preds)):
                # compute loss on one data point
                # print(f"x: {x}, y: {y}")
                response = y_pred
                logger.info(f"response: {response}")
                response.alias += f"_{i}"
                # TODO: when it is train, need to pass the data to be something used for eval.
                loss = self.loss_fn(
                    kwargs={
                        "y": response,
                        "y_gt": Parameter(
                            data=y,
                            role_desc="The ground truth",
                            requires_opt=False,
                            alias=f"y_{i}",
                        ),
                    }
                )
                loss.alias += f"_step_{steps}_batch_{i}"
                print(f"y_gt: {y})")
                losses.append(loss)
                # loss.draw_graph(filepath="loss1")
            # convert y_pred to value
            y_preds_data = [y_p.data for y_p in y_preds]
            batch_y_data = batch_y.tolist()
            print(f"y_preds_data: {y_preds_data}")
            print(f"batch_y: {batch_y_data}")
            acc, acc_list = self.evaluator.compute(y_preds_data, batch_y_data)
            # 1. Add constraint 1, only train when observe errors/loss > 0
            # loss = 1 - acc
            print(f"batch acc: {acc}")
            if acc == 1:
                print(f"no training loss, acc: {acc}")
                continue
            # resample the losses across batch
            for i, acc_i in enumerate(acc_list):
                if acc_i < 1:
                    errors_losses.append(losses[i])
                else:
                    correct_losses.append(losses[i])
            print(f"len(errors_losses): {len(errors_losses)}")
            print(f"len(correct_losses): {len(correct_losses)}")
            sampled_correct_losses = []
            sampled_errors_losses = []
            max_error_samples = 4
            if len(errors_losses) > 0:
                # sample 2 correct losses

                sampled_errors_losses = random.sample(
                    errors_losses, min(max_error_samples, len(errors_losses))
                )  # limit to 4
                print(f"sampling errors: {len(sampled_errors_losses)}")
                sampled_correct_losses = random.sample(
                    correct_losses, min(len(correct_losses), len(sampled_errors_losses))
                )
            # control the actual batch size for proposing
            print(f"len(sampled_errors_losses): {len(sampled_errors_losses)}")
            print(f"len(sampled_correct_losses): {len(sampled_correct_losses)}")
            total_loss = sum(sampled_errors_losses + sampled_correct_losses)
            # resampled_acc = len(sampled_correct_losses) / (
            #     len(sampled_correct_losses) + len(sampled_errors_losses)
            # )
            # compute the textual loss
            # TODO: need to observe a batch of data, such that we can see that it always undercount 1
            # total_loss = sum(losses)
            print("loss backward...")
            total_loss.backward()
            print("optimizer propose...")
            # 2. Propose and observe on the training set (and even add this in the history)
            for i in range(num_proposals):
                print(f"proposal: {i}")
                self.optimizer.propose()
                new_preds = self.train_batch_worker(batch_x)
                new_y_preds_data = [y_p.data for y_p in new_preds]
                new_batch_y_data = batch_y.tolist()
                new_acc = self.evaluator.compute(new_y_preds_data, new_batch_y_data)[0]
                if new_acc > acc:
                    print(f"new acc: {new_acc} > {acc}")
                    break
                else:
                    print(f"revert: {acc}")
                    self.optimizer.revert()
            if not self.optimizer.proposing:
                print(
                    "no proposal can improve the training accuracy, no need to validate"
                )
                # error still exists, no need to clean
                continue

            # now we get test acc
            prompts = self._get_param_values()
            print(f"new prompt: {prompts}")
            # total_loss.draw_graph(filepath=f"total_loss_step_{steps}")
            print("Start evaluate")

            # save_json(total_loss.to_dict(), "total_loss_adalflow.json")

            eval_acc, eval_acc_list = self.evaluate_dataset(
                dataset_type="val", max_samples=max_samples
            )
            print(f"val_acc: {eval_acc}, last acc: {results['val_acc'][-1]}")
            if eval_acc > results["val_acc"][-1]:
                print("optimizer step")
                self.optimizer.step()
                results["val_acc"].append(eval_acc)
                # error and correct signal will never be carried over
                errors_losses = []
                correct_losses = []

            else:
                self.optimizer.revert()
                print("optimizer revert")
                results["val_acc"].append(results["val_acc"][-1])
            final_prompts = self._get_param_values()
            results["prompts"].append(final_prompts)

            test_acc, test_acc_list = self.evaluate_dataset(
                dataset_type="test", max_samples=max_samples
            )
            results["test_acc"].append(test_acc)
            print(f"test_acc: {test_acc}")

            save_json(results, save_result_file_path)

    def fit_orpo(
        self,
        max_steps: int = 3,
        max_samples=20,
        results: Dict = None,
    ):
        self.task.train()
        max_steps = max_steps
        max_samples = max_samples
        task_name = self.task.__class__.__name__
        # num_proposals = 4
        save_result_file_path = f"results_adalflow_orpo_task_{task_name}_max_steps_{max_steps}_max_samples_{max_samples}.json"
        num_epochs = max_steps // len(self.train_loader) + 1
        total_step = 0
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for steps, (batch_x, batch_y) in enumerate(
                (pbar := tqdm(self.train_loader, position=0))
            ):
                total_step += 1
                pbar.set_description(f"Training Step: {steps}")
                if steps >= max_steps:
                    print(f"steps: {steps} >= max_steps: {max_steps}")
                    break

                # it does not use train batch yet
                # self.task.train()

                # y_preds = self.train_batch_worker(batch_x)
                self.orpo_optimizer.propose()
                prompts = self._get_param_values()
                print(f"new prompt: {prompts}")

                # validate
                val_acc, val_acc_list = self.evaluate_dataset(
                    dataset_type="val", max_samples=max_samples
                )
                if val_acc > results["val_acc"][-1]:
                    print(
                        f" optimizer step, val_acc: {val_acc} > {results['val_acc'][-1]}"
                    )
                    self.orpo_optimizer.step(score=val_acc)
                    results["val_acc"].append(val_acc)
                    results["prompts"].append(prompts)
                else:
                    print(
                        f"optimizer revert, val_acc: {val_acc} <= {results['val_acc'][-1]} "
                    )
                    self.orpo_optimizer.revert()
                    continue  # no need to test

                # test
                test_acc, test_acc_list = self.evaluate_dataset(
                    dataset_type="test", max_samples=max_samples
                )
                results["test_acc"].append(test_acc)
                print(f"test_acc: {test_acc}")
                # save the results
                save_json(results, save_result_file_path)

    @staticmethod
    def _compute_losses(batch_x, batch_y, y_preds, loss_fn, steps):
        losses: List[Parameter] = []
        for i, (x, y, y_pred) in enumerate(zip(batch_x, batch_y, y_preds)):
            # compute loss on one data point
            # print(f"x: {x}, y: {y}")
            response = y_pred
            logger.info(f"response: {response}")
            response.alias += f"_{i}"
            # TODO: when it is train, need to pass the data to be something used for eval.
            loss = loss_fn(
                kwargs={
                    "y": response,
                    "y_gt": Parameter(
                        data=y,
                        role_desc="The ground truth",
                        requires_opt=False,
                        alias=f"y_{i}",
                    ),
                }
            )
            loss.alias += f"_step_{steps}_batch_{i}"
            print(f"y_gt: {y})")
            losses.append(loss)
        return losses

    def fit_v3(
        self,
        max_steps: int = 3,
        max_samples=20,
        results: Dict = None,
        optimizer: TextualGradientDescent = None,
        optimizer_type: str = "tgd",
    ):
        # TODO: save a best prompt or top 2
        self.task.train()
        optimizer.zero_grad()
        logger.info(f"Training started: {self.task.training}")
        max_steps = max_steps
        max_samples = max_samples
        task_name = self.task.__class__.__name__
        num_proposals = 6
        save_result_file_path = f"results_adalflow_v3_optimizer_{optimizer.__class__.__name__}_task_{task_name}_max_steps_{max_steps}_max_samples_{max_samples}.json"
        # TODO: compute the epoch based on the number of samples
        # errors_losses: List[Parameter] = []
        # correct_losses: List[Parameter] = []
        all_x = []
        all_y = []
        all_losses = []
        all_y_preds = []

        # TODO: deduplicate, use set all_x and all_y, they might become too big

        # estimate the epich size with the steps
        num_epochs = max_steps // len(self.train_loader) + 1
        total_step = 0
        for epoch in tqdm(range(num_epochs), desc="Epoch"):

            print(f"epoch: {epoch}")

            for steps, (batch_x, batch_y) in enumerate(
                (pbar := tqdm(self.train_loader, position=0))
            ):
                total_step += 1
                pbar.set_description(f"Training Step: {steps}")
                if steps >= max_steps:
                    print(f"steps: {steps} >= max_steps: {max_steps}")
                    break
                self.task.train()

                y_preds = self.train_batch_worker(
                    batch_x
                )  # generator should always guarentee data even if it gives error
                # compute loss each data point
                losses = []
                if optimizer_type == "tgd":
                    losses = self._compute_losses(
                        batch_x, batch_y, y_preds, self.loss_fn, steps
                    )

                # loss.draw_graph(filepath="loss1")
                # convert y_pred to value
                y_preds_data = [y_p.data for y_p in y_preds]
                batch_y_data = batch_y.tolist()
                print(f"y_preds_data: {y_preds_data}")
                print(f"batch_y: {batch_y_data}")
                acc, acc_list = self.evaluator.compute(y_preds_data, batch_y_data)
                # 1. Add constraint 1, only train when observe errors/loss > 0
                # loss = 1 - acc
                print(f"batch acc: {acc}")
                # if acc == 1:
                #     print(f"no training loss, acc: {acc}")
                #     continue
                # gather the data to the last step
                all_x.extend(batch_x)
                all_y.extend(batch_y.tolist())
                all_losses.extend(losses)
                all_y_preds.extend(y_preds_data)
                all_acc, all_acc_list = self.evaluator.compute(all_y_preds, all_y)
                max_error_samples = 4
                max_correct_samples = 4
                # NOTE: the real batch size is 8 for the loss.
                print(f"all_acc: {all_acc}, all_acc_list: {all_acc_list}")
                correct_indices = [i for i, acc in enumerate(all_acc_list) if acc == 1]
                error_indices = [i for i, acc in enumerate(all_acc_list) if acc == 0]
                if len(error_indices) == 0:
                    print(f"no errors so far, acc: {all_acc}")
                    continue
                print(f"len(error_indices): {len(error_indices)}")
                print(f"len(correct_indices): {len(correct_indices)}")

                # Sample up to four indices from both correct and error lists
                # NOTE: it is important to make the subset has a higher ratio of errors so that proposals can pass the pipeline
                sampled_error_indices = random.sample(
                    error_indices, min(max_error_samples, len(error_indices))
                )
                num_errors = len(sampled_error_indices)
                max_num_correct_samples = 2 * num_errors
                sampled_correct_indices = random.sample(
                    correct_indices,
                    min(
                        max_correct_samples,
                        max_num_correct_samples,
                        len(correct_indices),
                    ),
                )

                sampled_batch_x = [all_x[i] for i in sampled_error_indices] + [
                    all_x[i] for i in sampled_correct_indices
                ]
                sampled_batch_y = [all_y[i] for i in sampled_error_indices] + [
                    all_y[i] for i in sampled_correct_indices
                ]
                sampled_y_preds = [all_y_preds[i] for i in sampled_error_indices] + [
                    all_y_preds[i] for i in sampled_correct_indices
                ]
                sample_acc = self.evaluator.compute(sampled_y_preds, sampled_batch_y)[0]

                print(f"len(sampled_errors_losses): {len(sampled_error_indices)}")
                print(f"len(sampled_correct_losses): {len(sampled_correct_indices)}")

                # compute the textual loss
                # TODO: need to observe a batch of data, such that we can see that it always undercount 1
                # total_loss = sum(losses)
                print("loss backward...")
                if optimizer_type == "tgd":
                    # now resample the correct and errors
                    total_loss = [all_losses[i] for i in sampled_error_indices] + [
                        all_losses[i] for i in sampled_correct_indices
                    ]
                    total_loss = sum(total_loss)
                    total_loss.backward()
                print("optimizer propose...")
                # 2. Propose and observe on the training set (and even add this in the history)
                for i in range(num_proposals):
                    print(f"proposal: {i}")
                    if optimizer_type == "tgd":
                        optimizer.propose()
                    elif optimizer_type == "orpo":  # TODO: add raw response
                        training_samples: List[str] = [
                            f"{x}\nPrediction: {y_pred},\n Correct Answer: {y_gt}"
                            for x, y_pred, y_gt in zip(
                                sampled_batch_x, sampled_y_preds, sampled_batch_y
                            )
                        ]
                        optimizer.propose(training_samples)
                    else:
                        raise ValueError(
                            f"Optimizer type: {optimizer_type} not supported"
                        )
                    new_preds = self.train_batch_worker(sampled_batch_x)
                    new_y_preds_data = [y_p.data for y_p in new_preds]
                    new_acc = self.evaluator.compute(new_y_preds_data, sampled_batch_y)[
                        0
                    ]
                    if new_acc > sample_acc:
                        print(
                            f"Pass the subset check, new acc: {new_acc} > {sample_acc}"
                        )
                    else:
                        print(
                            f"Failed the subset check, revert: {new_acc} <= {sample_acc}"
                        )
                        optimizer.revert()
                        continue
                    new_preds = self.train_batch_worker(all_x)
                    new_y_preds_data = [y_p.data for y_p in new_preds]
                    new_acc = self.evaluator.compute(new_y_preds_data, all_y)[0]
                    if new_acc > all_acc:
                        print(
                            f"Pass the whole set check, new acc: {new_acc} > {all_acc}"
                        )
                        break
                    else:
                        print(
                            f"Fail the whole set check, revert: {new_acc} <= {all_acc}"
                        )
                        # optimizer.revert()
                        continue
                if not optimizer.proposing:
                    print(
                        "no proposal can improve the training accuracy, Will try next batch"
                    )
                    # error still exists, no need to clean
                    continue

                # now we get test acc
                prompts = self._get_param_values()
                print(f"new prompt: {prompts}")
                # total_loss.draw_graph(filepath=f"total_loss_step_{steps}")
                print("Start evaluate")

                # save_json(total_loss.to_dict(), "total_loss_adalflow.json")

                eval_acc, eval_acc_list = self.evaluate_dataset(
                    dataset_type="val", max_samples=max_samples
                )
                print(f"val_acc: {eval_acc}, last acc: {results['val_acc'][-1]}")
                if eval_acc > results["val_acc"][-1]:
                    print(
                        f"Pass the val set check, optimizer step, {eval_acc} > {results['val_acc'][-1]}"
                    )
                    if optimizer_type == "tgd":
                        optimizer.step()
                    elif optimizer_type == "orpo":
                        optimizer.step(score=eval_acc)
                    else:
                        raise ValueError(
                            f"Optimizer type: {optimizer_type} not supported"
                        )
                    results["val_acc"].append(eval_acc)
                    # error and correct signal will never be carried over
                    # errors_losses = []
                    # correct_losses = []
                    all_x = []
                    all_y = []
                    all_losses = []
                    all_y_preds = []

                else:
                    optimizer.revert()
                    print(
                        f"Fail the val set check, optimizer revert, {eval_acc} <= {results['val_acc'][-1]}"
                    )
                    continue
                    # results["val_acc"].append(results["val_acc"][-1])
                final_prompts = self._get_param_values()
                results["prompts"].append(final_prompts)

                test_acc, test_acc_list = self.evaluate_dataset(
                    dataset_type="test", max_samples=max_samples
                )
                results["test_acc"].append(test_acc)
                print(f"test_acc: {test_acc}")

                save_json(results, save_result_file_path)

    # def eval_no_concurrent(self, dataset=None, max_samples: int = 100):
    #     if dataset is None:
    #         print("No dataset provided, using test set")
    #         dataset = self.test_set

    #     # set it to eval mode
    #     self.training = False
    #     x, y, y_pred = [], [], []
    #     tqdm_loader = tqdm(dataset)
    #     for _, sample in enumerate(tqdm_loader):
    #         y.append(sample[1])
    #         y_pred.append(self.task.call(question=sample[0]))
    #         x.append(sample[0])
    #         print(f"y: {y}, y_pred: {y_pred}, x: {x}")
    #         tqdm_loader.set_description(
    #             f"Accuracy: {self.evaluator.compute(y_pred, y)}"
    #         )

    #     return self.evaluator.compute(y_pred, y)[1]

    def train_batch_worker(self, batch_x, max_workers: int = 4):
        y_preds = []
        self.task.train()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for _, sample in enumerate(batch_x):
                future = executor.submit(self.task.call, question=sample)
                futures.append((future, sample))
            for future, sample in futures:
                y_preds.append(future.result())
        return y_preds

    def evaluate_dataset(
        self, dataset_type: str = "test", max_samples: int = 100, num_workers: int = 4
    ):

        # set it to eval mode
        dataset = None
        if dataset_type == "test":
            dataset = self.test_set
        elif dataset_type == "val":
            dataset = self.val_set
        elif dataset_type == "train":
            dataset = self.train_set
        else:
            raise ValueError(f"dataset_type: {dataset_type} not supported")

        self.task.eval()
        logger.debug(
            f"{self.__class__.__name__}: trainer eval stage on {dataset_type} dataset"
        )
        x, y, y_pred = [], [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
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
                    f"{dataset_type} Accuracy: {self.evaluator.compute(y_pred, y)[0]}"
                )
                # print(f"y: {y}, y_pred: {y_pred}, x: {x}")
        return self.evaluator.compute(y_pred, y)  # acc and acc_list

    def _extra_repr(self) -> str:
        s = f"train_set: {len(self.train_set)}, val_set: {len(self.val_set)}, test_set: {len(self.test_set)}, "
        s += f"eval_fn: {self.eval_fn}, "
        s += f"evaluator: {self.evaluator}"
        return s


# TODO: implement cache for generator(make it configurable)
if __name__ == "__main__":
    # task = ObjectCountTask(**gpt_3_model)
    # # logger = get_logger(level="DEBUG")
    # print(task)
    # exit(0)
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

    trainer = ObjectCountTrainer(
        task_model_config=gpt_3_model,
        backward_engine_model_config=gpt_3_model,
        tgd_model_config=gpt_4o_model,
    )
    # print(trainer)
    # max_samples = 100
    # max_steps = 10
    # optimizer = trainer.optimizer
    # optimizer_type = "tgd"
    # # optimizer = trainer.orpo_optimizer
    # # optimizer_type = "orpo"

    # test_acc, test_acc_list = trainer.evaluate_dataset(
    #     dataset_type="test", max_samples=max_samples
    # )
    # print(f"test_acc: {test_acc}")
    # val_acc, val_acc_list = trainer.evaluate_dataset(
    #     dataset_type="val", max_samples=max_samples
    # )
    # results = {
    #     "val_acc": [val_acc],
    #     "test_acc": [test_acc],
    #     "prompts": [trainer._get_param_values()],
    # }
    # print(f"val_acc: {val_acc}")
    # # trainer.fit_orpo(max_samples=max_samples, results=results, max_steps=max_steps)
    # trainer.fit_v3(
    #     max_samples=max_samples,
    #     results=results,
    #     max_steps=max_steps,
    #     optimizer=optimizer,
    #     optimizer_type=optimizer_type,
    # )

    # test the new trainer
    train_object_count_text_grad_v1(
        batch_size=4,
        max_steps=5,
        max_samples=100,
        num_workers=4,
        strategy="constrained",
    )
    # import torch

    # torch.cat
    # test_acc, test_acc_list = trainer.evaluate_dataset(
    #     dataset_type="test", max_samples=None
    # )
    # print(f"test_acc after optimization: {test_acc}")
    # TODO: use cache for the generator
    #
    # output = trainer.eval(dataset=trainer.val_set, max_samples=5)
    # print(f"eval output: {output}")
    # gpt-3.5-turbo test 0.69 [10 samples = 0.8], 0.72 (simple pasing, instead of json)
    # 0.73 with new parameters close to text-grad, using separate prompt: 0.81
    # single prompt without you: -> 0.82 <SYSTEM> system prompt.<SYS>0.78 <START_OF_SYSTEM_PROMPT> system prompt.<END_OF_SYSTEM_PROMPT> =>0.84 json_output = 0.68
    # yaml parser = 0.73  # json fixed 0.8 with different field description
    # text/ user role -> 0.76
    # so there is performance drop if we put the same prompt together
    # gpt-4o test 0.94

    # eval: 0.8
    # trainer.train(max_epochs=1)
