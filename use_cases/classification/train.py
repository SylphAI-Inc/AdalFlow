from use_cases.classification.task import TRECClassifier
from use_cases.classification.eval import ClassifierEvaluator
from core.component import Component
from use_cases.classification.data import (
    TrecDataset,
    ToSampleStr,
    SamplesToStr,
    dataset,
    _COARSE_LABELS_DESC,
    _COARSE_LABELS,
    _FINE_LABELS,
)
from torch.utils.data import DataLoader
import random


from typing import Any, Optional, Sequence, Dict
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, RandomSampler


class Orchestrator(Component):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._example_input = "What is the capital of France?"

    @property
    def example_input(self):
        return "How did serfdom develop in and then leave Russia ?"

    @example_input.setter
    def example_input(self, value):
        self._example_input = value

    # def training_step(self, *args, **kwargs) -> None:
    #     raise NotImplementedError("training_step method is not implemented")

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("train method is not implemented")

    def _extra_repr(self) -> str:
        return super()._extra_repr() + f"example_input={self._example_input}"


from torch.utils.data import Sampler  # better used in data loader
from typing import List, Sequence
import math


# class ClassSampler(Component):
#     def __init__(self, dataset, num_classes: int):
#         super().__init__()
#         self.dataset = dataset
#         # self.incides = [
#         #     i for i, data in enumerate(dataset) if data["class_index"] == class_index
#         # ]
#         num_classes = len(_COARSE_LABELS)
#         self.class_indices: List[List] = [[] for _ in range(num_classes)]
#         print(f"len of class_indices: {len(self.class_indices)}")
#         for i, data in enumerate(dataset):
#             self.class_indices[data["coarse_label"]].append(i)

#     def sample(self, shots: int, class_index: int) -> Sequence[str]:
#         indices = random.sample(self.class_indices[class_index], shots)
#         return [self.dataset[i] for i in indices]


# class ExampleOptimizer(Component):
#     def __init__(self, dataset, num_shots: int) -> None:
#         super().__init__()
#         self.dataset = dataset
#         self.num_classes = len(_COARSE_LABELS)
#         self.class_sampler = ClassSampler(self.dataset, self.num_classes)

#         # self.sampler = RandomSampler(self.dataset, replacement=False, num_samples=num_shots)

#     def random_sample(self, shots: int) -> Sequence[str]:
#         indices = random.sample(range(len(self.dataset)), shots)
#         return [self.dataset[i] for i in indices]

#     def random_sample_class_balanced(self, shots: int) -> Sequence[str]:
#         samples = []
#         samples_per_class = math.ceil(shots / self.num_classes)
#         print(f"samples_per_class: {samples_per_class}")
#         for class_index in range(self.num_classes):
#             samples.extend(self.class_sampler.sample(samples_per_class, class_index))
#         if len(samples) > shots:
#             # randomly sample the remaining samples
#             samples = random.sample(samples, shots)
#         return samples

# sampler = RandomSampler(self.dataset, replacement=False, num_samples=shots)
# iter_times = shots // len(self.dataset)
# samples = []
# # use sampler.next() to get the next sample
# for i in range(iter_times):
#     samples.append(self.sampler.next())


# its search and auto reasoning
# 0.95, F1: 0.911
# these examples are even better/
examples_str_dspy = r"""
---

Question: How many miles is it from NY to Austria ?

thought: Let's think step by step in order to produce the class_index. We see that the question is asking for a numeric value, specifically the distance in miles between two locations.

class_name: NUM

class_index: 5

---

Question: Where is Los Vegas ?

thought: Let's think step by step in order to produce the class_index. We see that the question is asking for the location of a place, which falls under the LOC class.

class_name: LOC

class_index: 4

---

Question: Who was Sherlock Holmes 's archenemy ?

thought: Let's think step by step in order to produce the class_index. We are looking for a specific person who is an archenemy of Sherlock Holmes, which falls under the category of human beings.

class_name: HUM

class_index: 3

---

Question: Who was Shakespeare 's Moorish general ?

thought: Let's think step by step in order to produce the class_index. We are looking for a specific person related to Shakespeare's works, so this question falls under the HUM class.

class_name: HUM

class_index: 3

---

Question: What type of exercise burns the most calories ?

thought: Let's think step by step in order to produce the class_index. We can see that the question is asking for a type of exercise, which falls under the category of a specific entity or concept related to physical activity.

class_name: Entity

class_index: 1
"""
# 0.9, F1: 0.838
examples_str_dspy_no_space = r"""
---
Question: How many miles is it from NY to Austria ?
thought: Let's think step by step in order to produce the class_index. We see that the question is asking for a numeric value, specifically the distance in miles between two locations.
class_name: NUM
class_index: 5
---
Question: Where is Los Vegas ?
thought: Let's think step by step in order to produce the class_index. We see that the question is asking for the location of a place, which falls under the LOC class.
class_name: LOC
class_index: 4
---
Question: Who was Sherlock Holmes 's archenemy ?
thought: Let's think step by step in order to produce the class_index. We are looking for a specific person who is an archenemy of Sherlock Holmes, which falls under the category of human beings.
class_name: HUM
class_index: 3
---
Question: Who was Shakespeare 's Moorish general ?
thought: Let's think step by step in order to produce the class_index. We are looking for a specific person related to Shakespeare's works, so this question falls under the HUM class.
class_name: HUM
class_index: 3
---
Question: What type of exercise burns the most calories ?
thought: Let's think step by step in order to produce the class_index. We can see that the question is asking for a type of exercise, which falls under the category of a specific entity or concept related to physical activity.
class_name: Entity
class_index: 1
"""
# 0.8, F1: 0.732
examples_str_dspy_no_thought = r"""Question: How many miles is it from NY to Austria ?
class_name: NUM
class_index: 5
-----
Question: Where is Los Vegas ?
class_name: LOC
class_index: 4
-----
Question: Who was Sherlock Holmes 's archenemy ?
class_name: HUM
class_index: 3
-----
Question: Who was Shakespeare 's Moorish general ?
class_name: HUM
class_index: 3
------
Question: What type of exercise burns the most calories ?
class_name: Entity
class_index: 1
------
"""
# 0.75, F1: 0.692
examples_str_dspy_no_thought1 = r"""Question: How many miles is it from NY to Austria ?
class_name: NUM
class_index: 5

Question: Where is Los Vegas ?
class_name: LOC
class_index: 4

Question: Who was Sherlock Holmes 's archenemy ?
class_name: HUM
class_index: 3

Question: Who was Shakespeare 's Moorish general ?
class_name: HUM
class_index: 3

Question: What type of exercise burns the most calories ?
class_name: Entity
class_index: 1
"""
# 0.8, 0.778
SYLPH_DEMO_STR = r"""Question: Who was credited with saying : `` I never met a man I did n't like '' ?
thought: This question refers to a specific person known for a famous quote. The correct class should reflect a category that involves individuals or persons.
class_name: Human being
class_index: 3

Question: What is ouzo?
thought: The question asks for a description or explanation of what 'ouzo' is, which likely relates to an abstract concept or description.
class_name: Description and abstract concept
class_index: 2

Question: Why are there 12 people on a jury in criminal cases?
thought: This question seeks an explanation for a legal procedure, thus it is related to an abstract concept regarding societal structures and functions.
class_name: Description and abstract concept
class_index: 2

Question: What is the U.S. location of Procter & Gamble corporate offices?
thought: This question asks for a specific geographical detail, the location of a corporate office, which fits into the category of locations.
class_name: Location
class_index: 4

Question: How does a scientific calculator work?
thought: The question requires an explanation of the mechanics or functionality of a scientific calculator, falling under descriptions and abstract concepts related to technology and tools.
class_name: Description and abstract concept
class_index: 2
"""

# 0.7, F1: 0.602
# sample at least one class per question [not just random sampling]
no_demo_str = r"""Question: Who was credited with saying : `` I never met a man I didn't like '' ?
class_name: Human being
class_index: 3

Question: What is ouzo?
class_name: Description and abstract concept
class_index: 2

Question: Why are there 12 people on a jury in criminal cases?
class_name: Description and abstract concept
class_index: 2

Question: What is the U.S. location of Procter & Gamble corporate offices?
class_name: Location
class_index: 4

Question: How does a scientific calculator work?
class_name: Description and abstract concept
class_index: 2
"""


# 0.9, F1: 0.865
no_question_STR = r"""thought: This question refers to a specific person known for a famous quote. The correct class should reflect a category that involves individuals or persons.
class_name: Human being
class_index: 3

thought: The question asks for a description or explanation of what 'ouzo' is, which likely relates to an abstract concept or description.
class_name: Description and abstract concept
class_index: 2

thought: This question seeks an explanation for a legal procedure, thus it is related to an abstract concept regarding societal structures and functions.
class_name: Description and abstract concept
class_index: 2

thought: This question asks for a specific geographical detail, the location of a corporate office, which fits into the category of locations.
class_name: Location
class_index: 4

thought: The question requires an explanation of the mechanics or functionality of a scientific calculator, falling under descriptions and abstract concepts related to technology and tools.
class_name: Description and abstract concept
class_index: 2
"""
# 0.85, F1: 0.821
no_question__dash_STR = r"""---------
thought: This question refers to a specific person known for a famous quote. The correct class should reflect a category that involves individuals or persons.
class_name: Human being
class_index: 3
---------
thought: The question asks for a description or explanation of what 'ouzo' is, which likely relates to an abstract concept or description.
class_name: Description and abstract concept
class_index: 2
---------
thought: This question seeks an explanation for a legal procedure, thus it is related to an abstract concept regarding societal structures and functions.
class_name: Description and abstract concept
class_index: 2
---------
thought: This question asks for a specific geographical detail, the location of a corporate office, which fits into the category of locations.
class_name: Location
class_index: 4
---------
thought: The question requires an explanation of the mechanics or functionality of a scientific calculator, falling under descriptions and abstract concepts related to technology and tools.
class_name: Description and abstract concept
class_index: 2
---------
"""
# 0.75, F1: 0.625
SYLPH_DEMO_STR_thought = r"""Question: Who was credited with saying : `` I never met a man I did n't like '' ?
thought: Let's think step by step in order to produce the class_index. We see that the question asks about a person known for a quote, indicating the need to identify an individual. Therefore, the suitable class is one that categorizes individuals.
class_name: Human being
class_index: 3

Question: What is ouzo?
thought: Let's think step by step in order to produce the class_index. We see that we're asked to define 'ouzo.' This question does not involve identifying a specific person or place but rather explaining what something is. Thus, it falls under a category that involves explanations or abstract concepts.
class_name: Description and abstract concept
class_index: 2

Question: Why are there 12 people on a jury in criminal cases?
thought: Let's think step by step in order to produce the class_index. We see that this question is about understanding a specific legal tradition, requiring a description of why a certain number is used in juries. It's more about explaining a concept than locating a place or identifying a person.
class_name: Description and abstract concept
class_index: 2

Question: What is the U.S. location of Procter & Gamble corporate offices?
thought: Let's think step by step in order to produce the class_index. We see that here, we are looking for a geographical answer, the location of a company's office. This information pertains directly to a place.
class_name: Location
class_index: 4

Question: How does a scientific calculator work?
thought: Let's think step by step in order to produce the class_index. We see that the query is about how a device functions, which involves explaining the technology and mechanics behind it. This is a detailed explanation, aligning it with descriptions and abstract concepts.
class_name: Description and abstract concept
class_index: 2
"""

random_class_balanced_str = r"""Question: What is the acronym for the rating system for air conditioner efficiency ?
class_name: Abbreviation 
class_index: 0
--------

Question: How do you say `` fresh '' in Spanish ?
class_name: Entity 
class_index: 1
--------

Question: What is proposition 98 about ?
class_name: Description and abstract concept 
class_index: 2
--------

Question: Who portrayed Sherlock Holmes in 14 films between 1939 and 1946 ?
class_name: Human being 
class_index: 3
--------

Question: Where is Hearst Castle , built by publisher William Randolph Hearst ?
class_name: Location 
class_index: 4
--------

Question: What year was the Avery Dennison company founded ?
class_name: Numeric value 
class_index: 5
--------
"""
random_class_balanced_str_with_thought = r"""Question: What is the acronym for the rating system for air conditioner efficiency ?
thought: The question asks for an abbreviation of a specific rating system, making it appropriate to classify under 'Abbreviation'.
class_name: Abbreviation 
class_index: 0
--------

Question: How do you say `` fresh '' in Spanish ?
thought: This question seeks the translation of the word "fresh" into Spanish, fitting under 'Entity' since it involves the name of an object or concept in another language.
class_name: Entity 
class_index: 1
--------

Question: What is proposition 98 about ?
thought: The question inquires about the content or purpose of a specific proposition, dealing with a description or abstract concept.
class_name: Description and abstract concept 
class_index: 2
--------

Question: Who portrayed Sherlock Holmes in 14 films between 1939 and 1946 ?
thought: This question asks for the identification of a person, specifically an actor, hence it is classified under 'Human being'.
class_name: Human being 
class_index: 3
--------

Question: Where is Hearst Castle , built by publisher William Randolph Hearst ?
thought: The question asks for the location of a specific landmark, Hearst Castle, which makes it fall under the 'Location' category.
class_name: Location 
class_index: 4
--------

Question: What year was the Avery Dennison company founded ?
thought: The question is asking for a specific year, a numeric value, when a company was founded.
class_name: Numeric value 
class_index: 5
--------
"""

random_class_balanced_str_with_thought_space = r"""Question: What is the acronym for the rating system for air conditioner efficiency ?

thought: The question asks for an abbreviation of a specific rating system, making it appropriate to classify under 'Abbreviation'.

class_name: Abbreviation 

class_index: 0

--------

Question: How do you say `` fresh '' in Spanish ?

thought: This question seeks the translation of the word "fresh" into Spanish, fitting under 'Entity' since it involves the name of an object or concept in another language.

class_name: Entity 

class_index: 1

--------

Question: What is proposition 98 about ?

thought: The question inquires about the content or purpose of a specific proposition, dealing with a description or abstract concept.

class_name: Description and abstract concept 

class_index: 2

--------

Question: Who portrayed Sherlock Holmes in 14 films between 1939 and 1946 ?

thought: This question asks for the identification of a person, specifically an actor, hence it is classified under 'Human being'.

class_name: Human being 

class_index: 3

--------

Question: Where is Hearst Castle , built by publisher William Randolph Hearst ?

thought: The question asks for the location of a specific landmark, Hearst Castle, which makes it fall under the 'Location' category.

class_name: Location 

class_index: 4

--------

Question: What year was the Avery Dennison company founded ?

thought: The question is asking for a specific year, a numeric value, when a company was founded.

class_name: Numeric value 

class_index: 5

--------
"""

random_class_balanced_str_with_thought_no_question = r"""thought: The question asks for an abbreviation of a specific rating system, making it appropriate to classify under 'Abbreviation'.
class_name: Abbreviation 
class_index: 0
--------
thought: This question seeks the translation of the word "fresh" into Spanish, fitting under 'Entity' since it involves the name of an object or concept in another language.
class_name: Entity 
class_index: 1
--------
thought: The question inquires about the content or purpose of a specific proposition, dealing with a description or abstract concept.
class_name: Description and abstract concept 
class_index: 2
--------
thought: This question asks for the identification of a person, specifically an actor, hence it is classified under 'Human being'.
class_name: Human being 
class_index: 3
--------
thought: The question asks for the location of a specific landmark, Hearst Castle, which makes it fall under the 'Location' category.
class_name: Location 
class_index: 4
--------
thought: The question is asking for a specific year, a numeric value, when a company was founded.
class_name: Numeric value 
class_index: 5
--------
"""
# 0.85, F1: 0.724
random_class_balanced_str_with_thought_no_question_with_space = r"""thought: The question asks for an abbreviation of a specific rating system, making it appropriate to classify under 'Abbreviation'.

class_name: Abbreviation 

class_index: 0
---
thought: This question seeks the translation of the word "fresh" into Spanish, fitting under 'Entity' since it involves the name of an object or concept in another language.

class_name: Entity 

class_index: 1
---
thought: The question inquires about the content or purpose of a specific proposition, dealing with a description or abstract concept.

class_name: Description and abstract concept 

class_index: 2
---
thought: This question asks for the identification of a person, specifically an actor, hence it is classified under 'Human being'.

class_name: Human being 

class_index: 3
---
thought: The question asks for the location of a specific landmark, Hearst Castle, which makes it fall under the 'Location' category.

class_name: Location 

class_index: 4
---
thought: The question is asking for a specific year, a numeric value, when a company was founded.

class_name: Numeric value 

class_index: 5
---
"""
from optimizer.optimizer import BootstrapFewShot
from optimizer.sampler import RandomSampler, ClassSampler
from typing import Tuple


# for this trainer, we will learn from pytorch lightning
class TrecTrainer(Orchestrator):
    r"""
    data loader which is random shuffed already, and the batch can be used as the # samples
    """

    def __init__(
        self,
        num_classes: int,
        train_dataset,
        eval_dataset,
        test_dataset=None,
        num_shots: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.task = TRECClassifier(
            labels=_COARSE_LABELS, labels_desc=_COARSE_LABELS_DESC
        )
        self.example_input = "How did serfdom develop in and then leave Russia ?"
        self.num_shots = 8
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.data_loader = DataLoader(
            self.train_dataset, batch_size=self.num_shots, shuffle=True
        )
        self.eval_data_loader = DataLoader(
            self.eval_dataset, batch_size=1, shuffle=False
        )  # use this for speeded up evaluation
        # self.sample_optimizer = ExampleOptimizer(self.train_dataset, self.num_shots)
        self.evaluator = ClassifierEvaluator(num_classes=self.num_classes)
        # self.to_sample_str = ToSampleStr()
        self.samples_to_str = SamplesToStr()

        self.params = dict(self.task.generator.named_parameters())
        print(f"params: {self.params}")

        self.sampler = RandomSampler(
            dataset=self.train_dataset, num_shots=self.num_shots
        )
        self.class_sampler = ClassSampler(
            self.train_dataset,
            self.num_classes,
            get_data_key_fun=lambda x: x["coarse_label"],
        )
        self.few_shot_optimizer = BootstrapFewShot(
            example_parameter=self.params["system_prompt.trainable_prompt_kwargs"],
            train_dataset=self.train_dataset,
            sampler=self.class_sampler,
            output_processors=self.samples_to_str,
        )

        print(f"few_shot_optimizer: {self.few_shot_optimizer}")
        print(f"few_shot_state_dict: {self.few_shot_optimizer.state_dict()}")

    # def random_shots(self, shots: int, class_balanced: bool = True) -> Sequence[str]:
    #     if class_balanced:
    #         samples = self.sample_optimizer.random_sample_class_balanced(shots)
    #     else:
    #         samples = self.sample_optimizer.random_sample(shots)
    #     # samples = self.sample_optimizer.random_sample(shots)
    #     samples_str = [self.to_sample_str(sample) for sample in samples]
    #     return samples_str

    def eval(self):
        r"""
        TODO: automatically tracking the average inference time
        """
        responses = []
        targets = []
        num_invalid = 0
        for data in self.eval_dataset.select(range(20)):
            print(f"data: {data}")
            task_input = data["text"]
            corse_label = data["coarse_label"]
            print(f"task_input: {task_input}, corse_label: {corse_label}")
            print(f"types: {type(task_input)}, {type(corse_label)}")

            response = self.task(task_input)
            if response == -1:
                print(f"invalid response: {response}")
                num_invalid += 1
                continue
            responses.append(response)
            targets.append(int(corse_label))

        # evaluate the responses
        print(f"responses: {responses}, targets: {targets}")
        print(f"num_invalid: {num_invalid}")
        accuracy, macro_f1_score = self.evaluator.run(responses, targets)
        return accuracy, macro_f1_score

    def batch_eval(self, batch: Dict[str, Any]) -> Tuple[float, float]:
        r"""
        batch evaluation
        """
        responses = []
        targets = []
        num_invalid = 0
        for text, corse_label in zip(batch["text"], batch["coarse_label"]):
            # print(f"data: {data}")
            task_input = text
            # corse_label = data["coarse_label"]
            print(f"task_input: {task_input}, corse_label: {corse_label}")
            print(f"types: {type(task_input)}, {type(corse_label)}")

            response = self.task(task_input)
            if response == -1:
                print(f"invalid response: {response}")
                num_invalid += 1
                continue
            responses.append(response)
            targets.append(int(corse_label))

        # evaluate the responses
        print(f"responses: {responses}, targets: {targets}")
        print(f"num_invalid: {num_invalid}")
        accuracy, macro_f1_score = self.evaluator.run(responses, targets)
        return accuracy, macro_f1_score

    def train(self, shots: int) -> None:
        r"""
        ICL with demonstrating examples, we might want to know the plot of the accuracy while using the few shots examples
        """
        # samples = self.sample_optimizer.random_sample(shots, self.train_dataset)
        # samples_str = self.random_shots(shots, True)
        # samples_str = [self.to_sample_str(sample) for sample in samples]
        # print(f"samples_str: {samples_str}")
        best_parameters = None
        best_eval = None
        best_score = 0
        max_steps = 4
        # for step in range(max_steps):
        for i, train_batch in enumerate(self.data_loader):
            if i >= max_steps:
                break
            print(f"step: {i}")
            print(f"train_batch: {train_batch}")
            # acc, macro_f1 = self.batch_eval(train_batch)
            # return
            samples_str = self.few_shot_optimizer.step(num_shots=shots)
            print(f"samples_str: {samples_str}")
            state_dict = {
                "generator": {"preset_prompt_kwargs": {"examples_str": samples_str}}
            }
            self.task.load_state_dict(state_dict)
            acc, macro_f1 = self.batch_eval(train_batch)  # should do batch evaluation
            print(f"Eval Accuracy: {acc}, F1: {macro_f1}")
            score = acc + macro_f1
            if score > best_score:
                best_score = score
                best_eval = (acc, macro_f1)
                best_parameters = state_dict
                print(f"best_score: {best_score}")
        print(f"best_parameters: {best_parameters}")
        print(f"best_eval: {best_eval}")

        # final evaluation
        acc, macro_f1 = self.eval()
        print(f"Eval Accuracy: {acc}, F1: {macro_f1}")

        # samples_str = random_class_balanced_str_with_thought_space
        # print(f"samples_str: {samples_str}")
        # # return
        # state_dict = {
        #     "generator": {"preset_prompt_kwargs": {"examples_str": samples_str}}
        # }
        # self.task.load_state_dict(state_dict)
        # self.task.generator.print_prompt()
        # acc, macro_f1 = self.eval()
        print(f"Eval Accuracy: {acc}, F1: {macro_f1}")


if __name__ == "__main__":
    import logging
    import sys

    # Configure logging to output to standard output (console)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # Example of setting logging to debug level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    trainer = TrecTrainer(
        num_classes=6, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    # print(trainer)
    trainer.train(6)
