DataClass
============

BaseDataClass
-------------
(BaseDataClass or just DataClass)

In PyTorch, 'Tensor' is the data container for all data as it is specifically designed for numerical data and can be used to communicate with 
`Module` and `Optimizer` class via wrapped into `Parameter` class.

In LLM applications, this is challenging as all types of data, text, list, dict, yaml, json etc, they can all
be used in components and be interacting with LLM via prompt, and get parsed back to different types of data too 
after the prediction.

Many existing library opt for Pydantic or marshmallow to define the data classes and to serialize and deserialize the data.
But, we build our BaseDataClass on the native `dataclasses` module in Python for the following reasons:

1. `dataclasses` save users time on writing the boilerplate code such as `__init__`, `__repr__`, `__eq__`, `__hash__`, `__str__` etc.
for the data classes but still is light and flexible, and it is more user-friendly than Pydantic.

2. Along with `field`, `default` and the `metadata` within field gives us ways to `describe` the data, set up the default value. 

Based on the needs of LLM applications, we provide ``BaseDataClass`` and it enhances the `dataclasses`:

Describe input or output data format to LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


What we want to let LLM know about our input/output data format:

**Description** of what this field is for.  We use `field` in `dataclasses` with `metadata` to describe the field with `desc` key.

Example: `thought: str = field(metadata={'desc': 'The reasoning or thought behind the question.'})`

**Required/Optional** whether this field is required or optional.
**Field Data Type** what type of data this field is.
**Order of the fields** matters as in a typical Chain of Thought, we want the reasoning/thought field to be output first, and then comes the answer.



How we achive this in `BaseDataClass`?

`Signature` and `Schema` (string) to describe either (1) the data class as input/output format or (2) the instance(data point) of the data class 
as a demonstration in LLM prompt.

`Schema` is more standard way in other library, often it only in the form of json string, here is one example:


Load data from dataset as example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Classmethods to load the data from a dataset, which is typically from Pytorch dataset or huggingface dataset and each data point is in
the form of a dictionary.


Compared with the major LLM frameworks, such as LlamaIndex, LangChain, LightRAG has the loosest data structure constraints.

BaseDataClass
-------------
Like the role of `Tensor` in `PyTorch`, `BaseDataClass` is the base class for all data classes in LightRAG. All data classes can potentially 
interact with the `Prompt` and `Generator` classes.

`to_dict`, `from_dict`, `to_json`, `from_json`, `to_file`, `from_file`.

Document
------------
We defined `Document` to function as a `string` container, and it can be used for any kind of text data along its `metadata` and relations
such as `parent_doc_id` if you have ever splitted the documents into chunks, and `embedding` if you have ever computed the embeddings for the document.

It functions as the data input type for some `string`-based components, such as `DocumentSplitter`, `Retriever`.