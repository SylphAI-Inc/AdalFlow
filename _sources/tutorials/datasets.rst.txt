.. _datasets:

Datasets
================

Datasets are wrapped in a :class:`Dataset<utils.data.Dataset>` object.
The `Dataset` often will be used together with :class:`utils.data.DataLoader` to load data in batches.
DataLoader can also handle parallel data loading with multiple workers and apply data shuffling.

To be able to use your data, you need to:

1. Create a subclass of :class:`DataClass<core.base_data_class.DataClass>` that defines the data structure, including a unique identifier, input and output fields for LLM calls.

2. Create a subclass of :class:`utils.data.Dataset` that defines how to load the data (local/cloud), split the data, and convert it to your defined DataClass, and how to load and preprocess the data. Optionally you can use PyTorch's dataset, the only thing is it often works with Tensor, you will need to convert it back to normal data at some point.

In default, AdalFlow saved any downloaded datasets in the `~/.adalflow/cached_datasets` directory.

You can see plenty of examples in the :ref:`apis-datasets` directory.
The examples of `DataClass` can be found at :ref:`datasets-types`.
