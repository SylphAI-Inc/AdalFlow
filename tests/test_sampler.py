import pytest
from copy import deepcopy
from typing import Any

from optim.sampler import ClassSampler, RandomSampler, Sample


@pytest.fixture
def mock_dataset():
    # Creating a dataset of 100 items, each a MagicMock with 'text' and 'label' attributes
    dataset = [{"text": f"Sample text {i}", "label": i % 10} for i in range(100)]
    return dataset


def get_key(item: Any):
    # Function to extract 'label' from the dataset item
    return item["label"]


class TestRandomSampler:
    def test_initialization(self, mock_dataset):
        sampler = RandomSampler(dataset=mock_dataset, default_num_shots=10)
        assert sampler.default_num_shots == 10  # Ensure num_shots is stored correctly

    def test_call_with_explicit_shots(self, mock_dataset):
        sampler = RandomSampler(dataset=mock_dataset)
        result = sampler.call(10)  # Call with explicit num_shots
        assert len(result) == 10  # Check if it returns exactly 10 samples
        assert all(
            item.data in mock_dataset for item in result
        )  # Ensure all items are from the dataset

    def test_call_with_default_shots(self, mock_dataset):
        # Create a sampler with a default number of shots
        sampler = RandomSampler(dataset=mock_dataset, default_num_shots=15)
        result = sampler(num_shots=10)  # Call without specifying num_shots
        assert len(result) == 10  # Should use the default num_shots
        assert all(item.data in mock_dataset for item in result)

    def test_call_without_num_shots_set(self, mock_dataset):
        # Expecting an error if num_shots is not set and not provided on call
        sampler = RandomSampler(dataset=mock_dataset)
        with pytest.raises(ValueError):
            result = sampler()  # No num_shots provided

    def test_random_replace_method(self, mock_dataset):
        sampler = RandomSampler(dataset=mock_dataset, default_num_shots=10)
        samples = sampler(10)
        replaced = sampler.random_replace(5, deepcopy(samples))
        assert len(replaced) == 10, "Number of samples should remain the same"

        # Ensure that the replaced samples are different from the original
        count_dif = sum(i != j for i, j in zip(samples, replaced))
        assert count_dif == 5, "Expected 5 replacements"


class TestClassSampler:
    def test_class_sampler_initialization(self, mock_dataset):
        sampler = ClassSampler(
            dataset=mock_dataset, num_classes=10, get_data_key_fun=get_key
        )
        assert len(sampler.class_indexces) == 10
        # Ensuring each class has the correct number of items assigned
        for idx, class_list in enumerate(sampler.class_indexces):
            assert all(mock_dataset[i]["label"] == idx for i in class_list)

    def test_sample_one_class(self, mock_dataset):
        sampler = ClassSampler(
            dataset=mock_dataset, num_classes=10, get_data_key_fun=get_key
        )
        samples = sampler._sample_one_class(5, 1)  # Sampling from class 1
        assert len(samples) == 5
        assert all(sample.data["label"] == 1 for sample in samples)

    def test_call_method_balances_classes(self, mock_dataset):
        sampler = ClassSampler(
            dataset=mock_dataset, num_classes=10, get_data_key_fun=get_key
        )
        result = sampler.call(50)  # Test total sampling
        assert len(result) == 50
        class_counts = {i: 0 for i in range(10)}
        for item in result:
            class_counts[item.data["label"]] += 1
        assert all(
            count == 5 for count in class_counts.values()
        )  # Check if fairly balanced

    def test_random_replace_method(self, mock_dataset):
        sampler = ClassSampler(
            dataset=mock_dataset, num_classes=10, get_data_key_fun=get_key
        )
        samples = sampler.call(50)
        replaced = sampler.random_replace(25, deepcopy(samples), replace=False)
        assert len(replaced) == 50, "Number of samples should remain the same"

        # Ensure that the replaced samples are different from the original
        count_dif = sum(i != j for i, j in zip(samples, replaced))
        assert count_dif == 25, "Expected 5 replacements"
