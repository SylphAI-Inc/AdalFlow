Introduction
=================

LLM-AutoDiff
-----------------
AdalFlow mainly relys on `LLM-AutoDiff <https://arxiv.org/abs/2501.16673>`__ to do automatic prompt engineering(APE).

Similar to Auto-differentiation in PyTorch, LLM-AutoDiff works by forming a runtime computation graph of prompts, hyperparameters, intermediate outputs, and losses in the `forward` pass.
In the `backward` pass, the `backward engine` LLM we put at each node will work together to identify which prompts are the cause of errors, so that a `feedback`-driven LM optimizer can leverage it to propose new prompts.


Components
-----------------
:class:`Component<core.component>` is to LM task pipelines what `nn.Module` is to PyTorch models.
A component can recursively contain and register other components, allow easy control of (1) `training` and `inference` modes, (2) visualization of the workflow structure, and (3) serialization and deserialization of the component.


We require a component to have (1) a `call` method, which will be called during the `inference` time,
and (2) a `forward` method, which will be called during the `training` time and output a `Parameter` object which has the output of the component wrapped in the `data` field.

There are four main types of components in AdalFlow:

1. `Component`: the base class of all components. With `forward` and `call` method, or `bicall`(handles both in one method). You use it to put together an LM workflow.
2. `GradComponent`: a subclass of `Component` that has a `backward` method. It defines a unit of computation that are capabalbe of backpropagation. One example is `Generator` and `Retriever`.
3. `DataComponent`: a subclass of `Component` that only has `call` method and does not handle any `Parameter` object. Examples include `Prompt`, `DataClassParser` which only handles the data formatting but rather the transformation.
4. `LossComponent`: a subclass of `Component` that likely takes an evaluation metric function and has a `backward` method. When it is attached to your LM workflow's output, the whole training pipeline is capable of backpropagation.

Parameters
-----------------
:class:`Parameter<optim.parameter.Parameter>` are used to save (1) intermediate forward data and (2) gradients/feedback, and (3) graph information such as `predecessors`.
It also has function like `draw_graph` to help you visualize the structure of your computation graph.

DataClass and Structured Output
----------------------------------
:class:`DataClass<core.base_data_class.DataClass>` is used for developers to define a data model.
Similar to `Pydantic`, it has methods like `to_yaml_signature`, `to_json_signature`, `to_yaml`, `to_json` to help you generate the data model schema and to generate the json/yaml data representation as strings.
It can be best used together with :class:`DataClassParser<components.output_parsers.dataclass_parser.DataClassParser>` for structured output.
