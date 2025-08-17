from dataclasses import dataclass, field
from typing import Literal
from adalflow.datasets.types import DataClass, BaseData
from adalflow.core import DataClass, required_field
from dataset import dataset
from adalflow.utils import setup_env 

setup_env()
@dataclass
class DataExtractionInput(BaseData):
    pages: list[str] = field(
        metadata={"desc": "The pages of the document"},
        default_factory=required_field()
    )
    date_rule: str = field(
        metadata={"desc": "The rule for extracting the date from the document"},
        default_factory=required_field()
    )
    expected_output: dict = field(
        metadata={"desc": "The ground truth of the data extraction"},
        default=None
    )


    __input_fields__ = ["pages", "date_rule"]


@dataclass
class ClientInformation(DataClass):

    primary_client_reasoning_scratch_pad: str = field(
        metadata={"desc": "The reasoning process for determining the primary client name. This is the name that should be used for the client information."},
        default_factory=required_field()
    )
    first_name_reasoning_scratch_pad: str = field(
        metadata={"desc": "The reasoning process for determining the client's first name."},
        default_factory=required_field()
    )
    first_name: str = field(
        metadata={"desc": "The client's given first name, parsed according to rules."},
        default_factory=required_field()
    )
    middle_name: str = field(
        metadata={"desc": "The client's middle name, if present as a full word after parsing."},
        default_factory=required_field()
    )
    last_name_reasoning_scratch_pad: str = field(
        metadata={"desc": "The reasoning process for determining the client's last name."},
        default_factory=required_field()
    )
    last_name: str = field(
        metadata={"desc": "The client's surname or family name, parsed according to rules."},
        default_factory=required_field()
    )


@dataclass
class DataExtractionOutput(BaseData):

    document_dates: list[str] = field(
        metadata={"desc": "The list of dates found in the document"},
        default_factory=list,
    )
    document_main_date: str = field(
        metadata={"desc": "The main date of the document, extracted from the list document_dates"},
        default_factory=required_field()
    )
    client_information: ClientInformation = field(
        metadata={"desc": "The client information of the document"},
        default_factory=required_field(),
    )

    __output_fields__ = ["document_dates", "document_main_date", "client_information"]

input_dataclass_list = []

for item in dataset:
    dataset_item = DataExtractionInput.from_dict(item)
    # dataset_item_1 = DataExtractionInput(
    #     pages=item["pages"],
    #     date_rule=item["date_rule"],
    #     expected_output=item["expected_output"]
    # )
    # assert dataset_item == dataset_item_1, "Dataset item does not match the expected structure"
    input_dataclass_list.append(dataset_item)


train_dataset = input_dataclass_list[:2]
val_dataset = input_dataclass_list[3:5]
test_dataset = input_dataclass_list[6:8]

len(train_dataset), len(val_dataset), len(test_dataset)

from adalflow.components.model_client.openai_client import OpenAIClient
import adalflow as adal

model_openai_o4_mini = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "o4-mini",  # or "o1"
        "reasoning": {
            "effort": "medium",  # low, medium, high
        }
    }
}

template = r"""
<START_OF_SYSTEM_MESSAGE>
    {{system_prompt}}

    {{output_format_str}}

    {% if few_shot_demos is not none %}
        Here are some examples:
        {{few_shot_demos}}
    {% endif %}
<END_OF_SYSTEM_MESSAGE>
<START_OF_USER_MESSAGE>
    {{input_str}}
<END_OF_USER_MESSAGE>
 """.strip()


task_prompt_document_extraction = r"""
You are a helpful assistant specialized in data processing and extraction.
""".strip()


from typing import Union, Optional
import adalflow as adal

class DataExtractor(adal.Component):

    def __init__(self, model_client: adal.ModelClient, model_kwargs: dict):
        super().__init__()

        # INPUT
        self.data_class_input = DataExtractionInput
        self.parser_input = adal.DataClassParser(
            data_class=self.data_class_input, return_data_class=True, format_type="json"
        )

        # OUTPUT
        task_desc_str = adal.Prompt(
            template=task_prompt_document_extraction,
            # prompt_kwargs={"classes": label_desc} #prompt variables to be hydrated
            )()
        self.data_class_output = DataExtractionOutput
        self.data_class_output.set_task_desc(task_desc_str)

        self.parser_output = adal.DataClassParser(
            data_class=self.data_class_output, return_data_class=True, format_type="json"
        )

        print(f"oututput format: {self.parser_output.get_output_format_str()}")

        # GENERATOR PARAMS
        prompt_kwargs = {
            "system_prompt": adal.Parameter(
                data=self.parser_output.get_task_desc_str(),
                role_desc="Task description",
                requires_opt=True,
                param_type=adal.ParameterType.PROMPT,
            ),
            "output_format_str": adal.Parameter(
                data=self.parser_output.get_output_format_str(),
                role_desc="Output format requirements",
                requires_opt=True,
                param_type=adal.ParameterType.PROMPT,
            ),

            # I didnt enable few shot demos to not overcomplicate, but it would be nice to get this working too. :D

            # "few_shot_demos": adal.Parameter(
            #     data=None,
            #     requires_opt=True,
            #     role_desc="Few shot examples to help the model",
            #     param_type=adal.ParameterType.DEMOS,
            # ),
        }

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            template=template,
            output_processors=self.parser_output,
            use_cache=False,
        )

        print(f"system prompt: {self.llm.get_prompt()}")

    def _prepare_input(self, dataset_item: DataExtractionInput):

        # QUESTION:
        # Im my use case, I have some arguments to pass to the prompt that are different for each dataset item.
        # Normaly I would put it in the system prompt, but here I was not sure if I could change the system prompt inside _prepare_input. Maybe it could break something.
        # So here in this example code, Im just passing it as a dump of the input data.

        input_data = self.data_class_input(pages=dataset_item.pages, date_rule=dataset_item.date_rule)
        input_str = self.parser_input.get_input_str(input_data)

        prompt_kwargs = {
            "input_str": adal.Parameter(
                data=input_str, requires_opt=False, role_desc="input to the LLM"
            )        }
        return prompt_kwargs

    def bicall(self,
        dataset_item: DataExtractionInput,
        id: Optional[str] = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        prompt_kwargs = self._prepare_input(dataset_item)
        output = self.llm(prompt_kwargs=prompt_kwargs, id=id)

        return output
    
task = DataExtractor(
    model_client=model_openai_o4_mini["model_client"],
    model_kwargs=model_openai_o4_mini["model_kwargs"],
)
print(task)

from typing import Dict, Callable, Any, Tuple

from adalflow.eval.answer_match_acc import AnswerMatchAcc

def eval_fn_data_extraction(
    y: DataExtractionOutput,
    y_gt: DataExtractionInput
) -> float:
    # I got some runs where the LLM failed to parse the output correctly.
    # TODO: try do to some retry somewhere in the code.

    # QUESTION:
    # I dont know if AdalFlow framework support it?
    # Maybe if the framework use the response format of the API it would increase the json output accuracy?

    try:
        patient_first_name_pred = y.client_information.first_name
        patient_first_name_gt = y_gt.expected_output["client_information"]["first_name"]
        return AnswerMatchAcc(type="exact_match").compute_single_item(patient_first_name_pred, patient_first_name_gt)

    except Exception as e:
        print(f"Parse error: {e}")
        return 0

class DataExtractorTrainner(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        teacher_model_config: Dict,
        backward_engine_model_config: Dict,
        text_optimizer_model_config: Dict,
    ):
        task = DataExtractor(model_client, model_kwargs)
        # eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        eval_fn = eval_fn_data_extraction
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0. When the LLM prediction failed with format parsing which results with errors, we set y_pred = -1",
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
            teacher_model_config=teacher_model_config,
        )

    def prepare_task(self, sample: DataExtractionInput):
        return self.task.call, {"dataset_item": sample, "id": sample.id}

    def prepare_eval(
        self, sample: DataExtractionInput, y_pred: adal.GeneratorOutput
    ) -> float:

        prediction = -1
        if y_pred and y_pred.data is not None:
            prediction = y_pred.data
        return self.eval_fn, {"y": prediction, "y_gt": sample}

    def prepare_loss(
        self, dataset_item: DataExtractionInput, y_pred: adal.Parameter, *args, **kwargs
    ) -> Tuple[Callable[..., Any], Dict]:

        # QUESTION:
        # How can I create a system that has multiple target variables to compute loss?


        full_response = y_pred.data
        y_label = -1  # default value for failed prediction
        try:
            first_name = full_response.data.client_information.first_name
        except Exception as e:
            print(f"Parse error: {e}")
            first_name = -1

        if isinstance(first_name, str) and first_name.strip():
            y_label = first_name.strip()

        y_pred.eval_input = y_label

        y_gt = adal.Parameter(
            name="y_gt",
            data=dataset_item.expected_output["client_information"]["first_name"],
            eval_input=dataset_item.expected_output["client_information"]["first_name"],
            requires_opt=False,
        )

        return self.loss_fn, {
            "kwargs": {
                "y": y_pred,
                "y_gt": y_gt
            },
            "id": dataset_item.id,
            "gt": y_gt.eval_input,
        }
    
def train(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
    train_batch_size=4,
    raw_shots: int = 0,
    bootstrap_shots: int = 1,
    max_steps=12,
    num_workers=4,
    strategy="constrained",
    optimization_order="sequential",
    debug=False,
):
    print("Starting training process...")

    # Define the model configuration for all components
    # gpt_4o_model = {
    #     "model_client": OpenAIClient(),
    #     "model_kwargs": {
    #         "model": "gpt-4o-mini",
    #         "temperature": 1,
    #         "top_p": 0.99,
    #         "max_tokens": 1000,
    #         # "frequency_penalty": 1,  # high for nto repeating prompt
    #     },
    # }
    model_openai_o4_mini = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "o4-mini",  # or "o1"
            "reasoning": {
                "effort": "medium",  # low, medium, high
            }
        }
    }

    model_openai_gpt_5 = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-5",  # or "o1"
            "reasoning": {
                "effort": "medium",  # low, medium, high
            }
        }
    }

    print(f"Component model configuration: {model_openai_o4_mini}")

    try:
        print("Initializing ADAL component...")
        adal_component = DataExtractorTrainner(
            model_client=model_client,
            model_kwargs=model_kwargs,
            text_optimizer_model_config=model_openai_gpt_5,
            backward_engine_model_config=model_openai_o4_mini,
            teacher_model_config=model_openai_o4_mini,
        )
        print("ADAL component initialized successfully")

        print("Initializing trainer...")
        trainer = adal.Trainer(
            train_batch_size=train_batch_size,
            adaltask=adal_component,
            strategy=strategy,
            max_steps=max_steps,
            num_workers=num_workers,
            raw_shots=raw_shots,
            bootstrap_shots=bootstrap_shots,
            debug=debug,
            weighted_sampling=True,
            optimization_order=optimization_order,
            exclude_input_fields_from_bootstrap_demos=True,
        )
        print("Trainer initialized successfully")

        print("Loading datasets...")
        # train_dataset, val_dataset, test_dataset = load_datasets()
        print(
            f"Datasets loaded - Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}"
        )

        print("Starting model training...")
        trainer.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            debug=debug,
        )
        print("Training completed successfully")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


model_openai_o4_mini = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "o4-mini",  # or "o1"
            "reasoning": {
                "effort": "medium",  # low, medium, high
            }
        }
    }

train(**model_openai_o4_mini)