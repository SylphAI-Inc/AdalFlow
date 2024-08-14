import adalflow as adal
from adalflow.datasets.types import TrecData
from adalflow.datasets.trec import _COARSE_LABELS_DESC, _COARSE_LABELS
from typing import Any, Callable, Dict, Tuple, Union, List, Optional
from dataclasses import dataclass, field

from adalflow.components.output_parsers.dataclass_parser import DataClassParser


task_desc_template = r"""You are a classifier. Given a question, you need to classify it into one of the following classes:
Format: class_index. class_name, class_description
{% if classes %}
{% for class in classes %}
{{loop.index-1}}. {{class.label}}, {{class.desc}}
{% endfor %}
{% endif %}
- Do not try to answer the question:
"""


@dataclass
class TRECExtendedData(TrecData):
    thought: str = field(
        metadata={
            "desc": "Your step-by-step reasoning to classify the question to class_name"
        },
        default=None,
    )
    __input_fields__ = ["question"]
    __output_fields__ = ["thought", "class_name", "class_index"]


class TRECClassifier(adal.Component):
    __doc__ = """We demonstrate how flexible the DataClass is to help use control dataformat, input field,output field,
    and their ordering in the formating."""

    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        label_desc = [
            {"label": label, "desc": desc}
            for label, desc in zip(_COARSE_LABELS, _COARSE_LABELS_DESC)
        ]

        task_desc_str = adal.Prompt(
            template=task_desc_template, prompt_kwargs={"classes": label_desc}
        )()

        self.data_class = TRECExtendedData

        yaml_parser = adal.YamlOutputParser(
            data_class=TRECExtendedData,
            include_fields=self.data_class.get_output_fields(),
            return_data_class=True,
        )

        prompt_kwargs = {
            "task_desc_str": task_desc_str,
            "output_format_str": yaml_parser.format_instructions(),
            "input_format_str": self.data_class.to_yaml_signature(
                include=self.data_class.get_input_fields()
            ),
        }

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            output_processors=yaml_parser,
        )

    def call(self, question: str, id: Optional[str] = None):
        input_data = self.data_class(question=question)
        input_str = input_data.to_yaml(include=["question"])
        prompt_kwargs = {
            "input_str": adal.Parameter(
                data=input_str, requires_opt=False, role_desc="input to the LLM"
            )
        }
        # self.llm.print_prompt(**prompt_kwargs)
        output = self.llm(prompt_kwargs)  # use forward method
        output.data.question = question
        return output


# when it failed to make the prediction. We should use label = -1
@adal.fun_to_component
def format_class_label(data: Optional[TrecData]) -> TrecData:
    if data is None:
        return TrecData(class_index=-1)
    return data


# Build a DAG


# # raw response, pred (final output) if passing failed, it will pass error to eval_fn,
# def compute_single_item_v2(pred: adal.GeneratorOutput, target: int) -> EvaluationResult:

#     feedback = ""
#     class_index = -1
#     if pred.data and pred.data.class_index is not None:
#         class_index = int(pred.data.class_index)
#     elif pred.error:
#         feedback += f"Error in prediction: {pred.error}"

#     if class_index < 0 or class_index >= len(_COARSE_LABELS):
#         feedback += f"Invalid class index: {class_index}"
#         return EvaluationResult(score=0.0, feedback=feedback)
#     if class_index == target:
#         return EvaluationResult(score=1.0, feedback=feedback)
#     feedback += f"Wrong prediction: {class_index} != {target}"
#     return EvaluationResult(score=0.0, feedback=feedback)


class TRECClassifierV2(adal.Component):

    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        label_desc = [
            {"label": label, "desc": desc}
            for label, desc in zip(_COARSE_LABELS, _COARSE_LABELS_DESC)
        ]

        task_desc_str = adal.Prompt(
            template=task_desc_template, prompt_kwargs={"classes": label_desc}
        )()

        self.data_class = TRECExtendedData
        self.data_class.set_task_desc(task_desc_str)

        self.parser = DataClassParser(
            data_class=self.data_class, return_data_class=True, format_type="json"
        )

        prompt_kwargs = {
            "task_desc_str": adal.Parameter(
                data=self.parser.get_task_desc_str(),
                role_desc="task description",
                # alias="task_desc",
                requires_opt=True,
            ),
            "output_format_str": adal.Parameter(
                data=self.parser.get_output_format_str(),
                role_desc="output format",
                # alias="output_format",
                requires_opt=False,
            ),
            # "examples_str": adal.Parameter(
            #     data=None,
            #     role_desc="examples",
            #     # alias="examples",
            #     param_type=adal.ParameterType.DEMOS,
            #     requires_opt=True,
            # ),
        }

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            output_processors=self.parser,
            use_cache=True,
        )

    def _prepare_input(self, question: str):
        input_data = self.data_class(question=question)
        input_str = self.parser.get_input_str(input_data)
        prompt_kwargs = {
            "input_str": adal.Parameter(
                data=input_str, requires_opt=False, role_desc="input to the LLM"
            )
        }
        return prompt_kwargs

    def call(
        self, question: str, id: Optional[str] = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        prompt_kwargs = self._prepare_input(question)
        output = self.llm(
            prompt_kwargs=prompt_kwargs, id=id
        )  # support both forward at training and call at inference if using __call__ method

        return output


template = r"""<START_OF_SYSTEM_MESSAGE>
{{system_prompt}}
<END_OF_SYSTEM_MESSAGE>
<START_OF_USER_MESSAGE>
{{input_str}}
<END_OF_USER_MESSAGE>
"""


# use one system prompt
class TRECClassifierV3(adal.Component):

    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        label_desc = [
            {"label": label, "desc": desc}
            for label, desc in zip(_COARSE_LABELS, _COARSE_LABELS_DESC)
        ]

        task_desc_str = adal.Prompt(
            template=task_desc_template, prompt_kwargs={"classes": label_desc}
        )()

        self.data_class = TRECExtendedData
        self.data_class.set_task_desc(task_desc_str)

        self.parser = DataClassParser(
            data_class=self.data_class, return_data_class=True, format_type="json"
        )

        prompt_kwargs = {
            "system_prompt": adal.Parameter(
                data=self.parser.get_task_desc_str()
                + "\n"
                + self.parser.get_output_format_str(),
                role_desc="Task description with output format requirements",
            ),
            # "examples_str": adal.Parameter(
            #     data=None,
            #     role_desc="examples",
            #     param_type=adal.ParameterType.DEMOS,
            #     requires_opt=True,
            # ),
        }

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            template=template,
            output_processors=self.parser,
            use_cache=True,
        )

    def _prepare_input(self, question: str):
        input_data = self.data_class(question=question)
        input_str = self.parser.get_input_str(input_data)
        prompt_kwargs = {
            "input_str": adal.Parameter(
                data=input_str, requires_opt=False, role_desc="input to the LLM"
            )
        }
        return prompt_kwargs

    def call(
        self, question: str, id: Optional[str] = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        prompt_kwargs = self._prepare_input(question)
        output = self.llm(
            prompt_kwargs=prompt_kwargs, id=id
        )  # support both forward at training and call at inference if using __call__ method

        return output


template = r"""<START_OF_SYSTEM_MESSAGE>
{{system_prompt}}
<END_OF_SYSTEM_MESSAGE>
<START_OF_USER_MESSAGE>
{{input_str}}
<END_OF_USER_MESSAGE>
"""

import re


# TODO: raw response should be passed to the eval_fn, so that we can even collect the format failure errors.
@adal.fun_to_component
def extract_class_index_value(text: str, get_feedback=False):
    pattern = re.compile(r"CLASS_INDEX\s*:\s*\$?(\d+)")

    match = pattern.search(text)

    if match:
        if get_feedback:
            return match.group(1), ""
        return match.group(1)
    else:  # process the failure
        print(f"No valid CLASS_INDEX: $VALUE found in the input text: {text}")
        feedback = "No valid CLASS_INDEX: $VALUE found"
        if get_feedback:
            return text, feedback
        return text


# @adal.fun_to_component
# def post_process_output(value: Optional[int] = None):
#     if value is None:
#         return -1
#     if value < 0 or value >= len(_COARSE_LABELS):
#         return -1
#     return value


# def compute_single_item(pred: int, target: int) -> float:
#     if pred == target:
#         return 1.0
#     return 0.0


# use one system prompt
# no structured output
class TRECClassifierV4(adal.Component):

    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        label_desc = [
            {"label": label, "desc": desc}
            for label, desc in zip(_COARSE_LABELS, _COARSE_LABELS_DESC)
        ]

        task_desc_str = adal.Prompt(
            template=task_desc_template, prompt_kwargs={"classes": label_desc}
        )()

        self.data_class = TRECExtendedData
        # self.data_class.set_task_desc(task_desc_str)

        self.parser = DataClassParser(
            data_class=self.data_class, return_data_class=True, format_type="yaml"
        )

        prompt_kwargs = {
            "system_prompt": adal.Parameter(
                data=task_desc_str
                + "\n"
                + "Think step by step. You MUST respond in format: 'CLASS_INDEX: $INT' where $INT is the class index you predict",
                role_desc="Task description with output format requirements",
            ),
            # "examples_str": adal.Parameter(
            #     data=None,
            #     role_desc="examples",
            #     param_type=adal.ParameterType.DEMOS,
            #     requires_opt=True,
            # ),
        }

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            template=template,
            output_processors=adal.Sequential(
                extract_class_index_value, adal.IntParser()
            ),
            use_cache=True,
        )

    def _prepare_input(self, question: str):
        input_data = self.data_class(question=question)
        input_str = self.parser.get_input_str(input_data)
        prompt_kwargs = {
            "input_str": adal.Parameter(
                data=input_str, requires_opt=False, role_desc="input to the LLM"
            )
        }
        return prompt_kwargs

    def call(
        self, question: str, id: Optional[str] = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        prompt_kwargs = self._prepare_input(question)
        output = self.llm(
            prompt_kwargs=prompt_kwargs, id=id
        )  # support both forward at training and call at inference if using __call__ method

        return output


from use_cases.classification.eval import ClassifierEvaluator


# train only few-shots
class TRECClassifierV2Trainable(adal.AdalComponent):

    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        teacher_model_config: Dict,
        optimizer_model_config: Dict = None,
        backward_engine_model_config: Dict = None,
    ):

        # label_desc = [
        #     {"class_index": i, "label": label, "desc": desc}
        #     for i, (label, desc) in enumerate(zip(_COARSE_LABELS, _COARSE_LABELS_DESC))
        # ]

        task = TRECClassifierV2(model_client, model_kwargs)
        evaluator = ClassifierEvaluator(num_classes=len(_COARSE_LABELS))
        eval_fn = evaluator.compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="accuracy: 1 if correct else 0",
        )
        self.teacher_model_config = teacher_model_config
        self.optimizer_model_config = optimizer_model_config
        self.backward_engine_model_config = backward_engine_model_config
        super().__init__(
            task=task,
            loss_fn=loss_fn,
            eval_fn=eval_fn,
            evaluator=evaluator,
        )
        print(f"After super init: self.loss_fn = {self.loss_fn}")

    def handle_one_task_sample(self, sample: TrecData):
        if self.loss_fn is None:
            raise ValueError("loss_fn is not initialized. It is None.")
        return self.task, {"question": sample.question, "id": sample.id}

    def handle_one_loss_sample(
        self, sample: TrecData, y_pred: adal.Parameter
    ) -> Tuple[Callable[..., Any], Dict]:
        if not isinstance(y_pred, adal.Parameter):
            raise ValueError(
                f"y_pred should be an instance of adal.Parameter, but got {type(y_pred)}"
            )
        # TODO: diferent parameters behave differently
        target_param = adal.Parameter(
            data=int(sample.class_index),
            requires_opt=False,
            role_desc="target class index",
        )
        target_param.set_eval_fn_input(sample.class_index)
        y_pred.set_eval_fn_input(-1)
        label_desc = [
            {"class_index": i, "label": label, "desc": desc}
            for i, (label, desc) in enumerate(zip(_COARSE_LABELS, _COARSE_LABELS_DESC))
        ]
        metadata = {"task_context": label_desc}
        # metadata = {}
        if (
            y_pred.full_response
            and y_pred.full_response.data
            and y_pred.full_response.data.class_index is not None
        ):
            y_pred.set_eval_fn_input(int(y_pred.full_response.data.class_index))
        else:
            y_pred.set_eval_fn_input(-1)
            if y_pred.full_response and y_pred.full_response.error:
                metadata["error"] = y_pred.full_response.error
        # print(f"y_pred: {y_pred.}, type: {type(y_pred.data)}")

        return self.loss_fn.forward, {
            "kwargs": {"pred": y_pred, "target": target_param},
            "metadata": metadata,
        }

    def evaluate_one_sample(
        self, sample: TrecData, y_pred: adal.GeneratorOutput, *args, **kwargs
    ) -> float:
        if not isinstance(y_pred, adal.GeneratorOutput):
            raise ValueError(
                f"y_pred should be an instance of adal.GeneratorOutput, but got {type(y_pred)}, {y_pred}"
            )
        try:
            label = y_pred.data.class_index
        except Exception as e:
            print(f"Error in getting the label: {e}, y_pred: {y_pred}")
            label = -1
        return self.eval_fn(label, int(sample.class_index))

    def configure_teacher_generator(self):
        super().configure_teacher_generator_helper(**self.teacher_model_config)

    def configure_backward_engine(self):
        return super().configure_backward_engine_helper(
            **self.backward_engine_model_config
        )

    def configure_optimizers(self) -> List[adal.Optimizer]:
        # when use
        do = super().configure_demo_optimizer_helper()
        self.configure_backward_engine()
        to = super().configure_text_optimizer_helper(**self.optimizer_model_config)
        return to + do


class TRECClassifierV4Trainable(adal.AdalComponent):

    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        teacher_model_config: Dict,
        optimizer_model_config: Dict = None,
        backward_engine_model_config: Dict = None,
    ):
        label_desc = [
            {"class_index": i, "label": label, "desc": desc}
            for i, (label, desc) in enumerate(zip(_COARSE_LABELS, _COARSE_LABELS_DESC))
        ]

        task = TRECClassifierV4(model_client, model_kwargs)
        eval_fn = compute_single_item  # noqa F821
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc=f"accuracy: 1 if correct else 0. task context: {label_desc}",
        )
        self.teacher_model_config = teacher_model_config
        self.optimizer_model_config = optimizer_model_config
        self.backward_engine_model_config = backward_engine_model_config
        super().__init__(
            task=task,
            loss_fn=loss_fn,
            eval_fn=eval_fn,
        )
        print(f"After super init: self.loss_fn = {self.loss_fn}")

    def handle_one_task_sample(self, sample: TrecData):
        if self.loss_fn is None:
            raise ValueError("loss_fn is not initialized. It is None.")
        return self.task, {"question": sample.question, "id": sample.id}

    # loss is a wrapper around eval_fn
    def handle_one_loss_sample(
        self, sample: TrecData, y_pred: adal.Parameter
    ) -> Tuple[Callable[..., Any], Dict]:
        if not isinstance(y_pred, adal.Parameter):
            raise ValueError(
                f"y_pred should be an instance of adal.Parameter, but got {type(y_pred)}"
            )
        # TODO: diferent parameters behave differently
        target_param = adal.Parameter(
            data=int(sample.class_index),
            requires_opt=False,
            role_desc="target class index",
        )
        target_param.set_eval_fn_input(sample.class_index)
        if y_pred.full_response:
            y_pred.set_eval_fn_input(y_pred.full_response.raw_response)
        else:
            raise ValueError(f"y_pred.full_response is None: {y_pred}")

        # print(f"y_pred: {y_pred.}, type: {type(y_pred.data)}")

        return self.loss_fn.forward, {
            "kwargs": {"pred": y_pred, "target": target_param}
        }

    # TODO: test evaluate one sample in the trainer
    def evaluate_one_sample(
        self, sample: TrecData, y_pred: adal.GeneratorOutput, *args, **kwargs
    ) -> float:
        if not isinstance(y_pred, adal.GeneratorOutput):
            raise ValueError(
                f"y_pred should be an instance of adal.GeneratorOutput, but got {type(y_pred)}, {y_pred}"
            )

        return self.eval_fn(y_pred.raw_response, int(sample.class_index)).score

    def configure_teacher_generator(self):
        return super().configure_teacher_generator_helper(**self.teacher_model_config)

    def configure_backward_engine(self):
        return super().configure_backward_engine_helper(
            **self.backward_engine_model_config
        )

    def configure_optimizers(self) -> List[adal.Optimizer]:
        # when use
        do = super().configure_demo_optimizer_helper()
        self.configure_backward_engine()
        to = super().configure_text_optimizer_helper(**self.optimizer_model_config)
        return to + do


from adalflow.datasets.trec import TrecDataset


def train(
    train_batch_size=4,
    max_steps=4,
    boostrap_shots=5,
    num_workers=4,
    raw_shots=0,
    strategy="random",
    model_client: adal.ModelClient = None,
    model_kwargs: Dict = None,
    teacher_model_config: Dict = None,
    optimizer_model_config: Dict = None,
    backward_engine_model_config: Dict = None,
    debug=False,
    weighted_sampling=False,
):

    # load data
    if debug:
        adal.get_logger(level="DEBUG")
    trainset = TrecDataset(split="train")
    valset = TrecDataset(split="val")
    testset = TrecDataset(split="test")
    trainer = adal.Trainer(
        adaltask=TRECClassifierV2Trainable(
            model_client=model_client,
            model_kwargs=model_kwargs,
            teacher_model_config=teacher_model_config,
            optimizer_model_config=optimizer_model_config,
            backward_engine_model_config=backward_engine_model_config,
        ),
        max_steps=max_steps,
        strategy=strategy,
        num_workers=num_workers,
        bootstrap_shots=boostrap_shots,
        raw_shots=raw_shots,
        train_batch_size=train_batch_size,
        debug=debug,
        weighted_sampling=weighted_sampling,
    )
    trainer.fit(train_dataset=trainset, val_dataset=valset, test_dataset=testset)


def trainv4(
    train_batch_size=4,
    max_steps=4,
    boostrap_shots=5,
    num_workers=4,
    raw_shots=0,
    strategy="random",
    model_client: adal.ModelClient = None,
    model_kwargs: Dict = None,
    teacher_model_config: Dict = None,
    optimizer_model_config: Dict = None,
    backward_engine_model_config: Dict = None,
    debug=False,
    weighted_sampling=False,
):

    # load data
    trainset = TrecDataset(split="train")
    valset = TrecDataset(split="val")
    testset = TrecDataset(split="test")
    trainer = adal.Trainer(
        adaltask=TRECClassifierV4Trainable(
            model_client=model_client,
            model_kwargs=model_kwargs,
            teacher_model_config=teacher_model_config,
            optimizer_model_config=optimizer_model_config,
            backward_engine_model_config=backward_engine_model_config,
        ),
        max_steps=max_steps,
        strategy=strategy,
        num_workers=num_workers,
        bootstrap_shots=boostrap_shots,
        raw_shots=raw_shots,
        train_batch_size=train_batch_size,
        debug=debug,
        weighted_sampling=weighted_sampling,
    )
    trainer.fit(train_dataset=trainset, val_dataset=valset, test_dataset=testset)


if __name__ == "__main__":

    adal.setup_env()
    from adalflow.components.model_client.openai_client import OpenAIClient

    # optimizer and teacher
    gpt_4o_model = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-4o",
            "temperature": 0.9,
            "top_p": 0.99,
        },
    }
    gpt_3_backward = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.9,
            "top_p": 0.99,
        },
    }

    from benchmarks.config import gpt_3_model, load_model

    trec_classifier = TRECClassifierV2(**load_model(**gpt_3_model))
    print(trec_classifier)

    question = "What does NASA stand for ?"
    response = trec_classifier(question)
    print(response)

    named_components_names = []
    for name, component in trec_classifier.named_components():
        named_components_names.append(name)

    named_grad_components_names = []
    for name, component in trec_classifier.named_components(grad_component_only=True):
        named_grad_components_names.append(name)

    named_parameters_names = []
    for name, parameter in trec_classifier.named_parameters():
        named_parameters_names.append(name)

    print(f"named components: {named_components_names}")
    print(f"named grad components: {named_grad_components_names}")
    print(f"named parameters: {named_parameters_names}")

    # test v4
    # trec_classifier_v4 = TRECClassifierV4(**load_model(**gpt_3_model))
    # print(trec_classifier_v4)
    # answer = trec_classifier_v4(question)
    # print(answer)

    # TODO: set resume from the last checkpoint
    # use gpt3.5 as backward engine and only gpt for optimizer
    train(
        **load_model(**gpt_3_model),
        teacher_model_config=gpt_4o_model,
        optimizer_model_config=gpt_4o_model,
        backward_engine_model_config=gpt_4o_model,
        debug=False,
        strategy="constrained",
        weighted_sampling=True,
        max_steps=8,
    )
