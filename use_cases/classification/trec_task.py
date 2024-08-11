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
                alias="task_desc",
                requires_opt=True,
            ),
            # "input_format_str": adal.Parameter(
            #     data=self.parser.get_input_format_str(),
            #     role_desc="input format",
            #     alias="input_format",
            # ),
            "output_format_str": adal.Parameter(
                data=self.parser.get_output_format_str(),
                role_desc="output format",
                alias="output_format",
                requires_opt=False,
            ),
            "examples_str": adal.Parameter(
                data=None,
                role_desc="examples",
                alias="examples",
                param_type=adal.ParameterType.DEMOS,
                requires_opt=False,
            ),
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


from use_cases.classification.eval import ClassifierEvaluator


# train only few-shots
class TRECClassifierV2Trainable(adal.AdalComponent):

    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        teacher_model_config: Dict,
    ):

        task = TRECClassifierV2(model_client, model_kwargs)
        evaluator = ClassifierEvaluator(num_classes=len(_COARSE_LABELS))
        eval_fn = evaluator.compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn, eval_fn_desc="accuracy: 1 if correct else 0"
        )
        self.teacher_model_config = teacher_model_config
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
            eval_input=sample.class_index,
            requires_opt=False,
            role_desc="target class index",
        )
        y_pred.eval_input = y_pred.full_response.data.class_index
        # print(f"y_pred: {y_pred.}, type: {type(y_pred.data)}")

        return self.loss_fn.forward, {
            "kwargs": {"pred": y_pred, "target": target_param}
        }

    def evaluate_one_sample(
        self, sample: TrecData, y_pred: adal.GeneratorOutput, *args, **kwargs
    ) -> float:
        if not isinstance(y_pred, adal.GeneratorOutput):
            raise ValueError(
                f"y_pred should be an instance of adal.GeneratorOutput, but got {type(y_pred)}, {y_pred}"
            )
        label = y_pred.data.class_index
        return self.eval_fn(int(label), int(sample.class_index))

    def configure_teacher_generator(self):
        return super().configure_teacher_generator_helper(**self.teacher_model_config)

    def configure_backward_engine(self):
        return super().configure_backward_engine_helper(**self.teacher_model_config)

    def configure_optimizers(self) -> List[adal.Optimizer]:
        # when use
        # do = super().configure_demo_optimizer_helper()
        self.configure_backward_engine()
        to = super().configure_text_optimizer_helper(**self.teacher_model_config)
        return to


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
    debug=False,
    weighted_sampling=False,
):

    # load data
    trainset = TrecDataset(split="train")
    valset = TrecDataset(split="val")
    testset = TrecDataset(split="test")
    trainer = adal.Trainer(
        adaltask=TRECClassifierV2Trainable(
            model_client=model_client,
            model_kwargs=model_kwargs,
            teacher_model_config=teacher_model_config,
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

    gpt_4o_model = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-4o",
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

    train(
        **load_model(**gpt_3_model),
        teacher_model_config=gpt_4o_model,
        debug=False,
        strategy="random",
        weighted_sampling=True,
    )
