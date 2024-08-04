"""Prepare the task pipeline"""

from typing import Any

from lightrag.core import Component, Generator
from lightrag.optim.parameter import Parameter, ParameterType
from lightrag.components.output_parsers import YamlOutputParser

from use_cases.question_answering.bhh_object_count.data import (
    parse_integer_answer,
    ObjectCountPredData,
)


class ObjectCountTaskOriginal(Component):
    r"""Same system prompt as text-grad paper, but with our one message prompt template, which has better starting performance"""

    def __init__(self, model_client, model_kwargs):
        super().__init__()

        template = """<START_OF_SYSTEM_PROMPT>{{system_prompt}}</END_OF_SYSTEM_PROMPT>{{input_str}}"""
        # 1. set up system prompt, and define the parameters for optimization.
        # NOTE: Dont use self.system_prompt
        # use self. will double the parameters, so we dont need that as we want the parameter to be part of the generator
        system_prompt = Parameter(
            alias="task_instruction",
            data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
            role_desc="To give task instruction to the language model in the system prompt",
            requires_opt=True,
            param_type=ParameterType.NONE,
        )
        self.llm_counter = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs={
                "system_prompt": system_prompt,
            },
            output_processors=parse_integer_answer,
            use_cache=True,
        )
        # TODO: make this data map function more robust (this is the final answer and the input to eval_fn)
        # self.llm_counter.set_data_map_func(lambda x: x.data.answer)
        print(f"llm_counter set_data_map_func, {self.llm_counter.data_map_func}")

    # TODO: the error will be a context
    def call(self, question: str, id: str = None) -> Any:  # Add id for tracing
        output = self.llm_counter(
            prompt_kwargs={"input_str": question, "id": id}
        )  # already support both training (forward + call)

        if not self.training:  # eval

            if output.data is None:
                print(f"Error in processing the question: {question}, output: {output}")
                output = -1
            else:
                output = int(output.data)
        return output


class ObjectCountTask(Component):
    r"""We will use (1) structured output (2) train both system prompt and the output formating prompt"""

    def __init__(self, model_client, model_kwargs):
        super().__init__()
        template = """<START_OF_SYSTEM_PROMPT>{{system_prompt}}<OUTPUT_FORMAT> {{output_format_str}}</OUTPUT_FORMAT></END_OF_SYSTEM_PROMPT>{{input_str}}"""
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
            # param_type=ParameterType.NONE,
        )
        instruction = "Do not change the fields in the JSON object. Only improve on the field descriptions."
        output_format_str = Parameter(
            alias="output_format",
            data="Respond with valid JSON object with the following schema:\n"
            + ObjectCountPredData.to_json_signature(),
            role_desc="To specify the LLM output format",
            instruction_to_optimizer=instruction,
            instruction_to_backward_engine=instruction,
            # param_type=ParameterType.PROMPT,
            requires_opt=True,
        )
        parser = YamlOutputParser(
            data_class=ObjectCountPredData, return_data_class=True
        )  # noqa: F841
        self.llm_counter = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs={
                "system_prompt": system_prompt,
                "output_format_str": output_format_str,
            },
            output_processors=parser,
        )
        # TODO: make this data map function more robust (this is the final answer and the input to eval_fn)
        self.llm_counter.set_data_map_func(lambda x: x.data.answer)
        print(f"llm_counter set_data_map_func, {self.llm_counter.data_map_func}")

    # TODO: the error will be a context
    def call(self, question: str, id: str = None) -> Any:
        output = self.llm_counter(
            prompt_kwargs={"input_str": question, "id": id}
        )  # already support both training (forward + call)

        if not self.training:  # eval

            if output.data is None:
                print(f"Error in processing the question: {question}, output: {output}")
                output = -1
            else:
                output = output.data.answer
        return output


if __name__ == "__main__":
    from use_cases.question_answering.bhh_object_count.config import gpt_3_model

    task = ObjectCountTask(**gpt_3_model)
    task_original = ObjectCountTaskOriginal(**gpt_3_model)

    question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"

    print(task(question))
    print(task_original(question))
