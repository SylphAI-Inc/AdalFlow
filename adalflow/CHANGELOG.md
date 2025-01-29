## [1.0.0.beta.1] - 2025-01-28
Updates mainly follow our first arxiv paper: [Auto-Differentiating Any LLM Workflow: A Farewell to Manual Prompting](https://arxiv.org/abs/2501.16673)

### Core Updates

#### ReAct Agent
- Added trainable task descriptions and docstrings
- Implemented `FunctionExpression` to `Function` conversion in planner responses
- Introduced `context_variables` for storage
- Added debug controls for output granularity
- Updated `__call__` to use `bicall` when available (dual train/eval modes)

#### Parser
- Migrated all parsers from `Component` to `Parser` class
- Deprecated `FunComponent` and `fun_to_component`
- Replaced with `FuncParser` and `func_to_parser`

#### Component
- Added `forward` and `bicall` methods
- Enhanced `FunctionTool` compatibility with default `__call__`
- Introduced `GradComponent` base class for training operations
- Added `fun_to_grad_component` decorator

#### Generator
- Integrated `BackwardPassSetup` into generator
- New engine features:
  - `disable_backward_engine` flag
  - `update_default_backward_pass_setup`
  - `_backward_through_all_predecessors`
- Added dedicated `BackwardEngine` class

#### Parameter
- Split into `Parameter` + `OutputParameter`
- Added tracking for:
  - `data_id` in training sets
  - `data_in_prompt` callable
  - `get_prompt_data()` method
- New visualization tools:
  - `draw_output_graph`
  - `draw_component_subgraph`
  - `draw_interactive_html_graph`

#### Training
- Added `LossComponent` for dedicated training mode
- Enhanced `tgd_optimizer` with:
  - `add_failed_proposal` (CH paper implementation)
  - One-parameter-at-a-time optimization
  - Improved prompt engineering for system awareness
- Updated `adal` with:
  - `loss_eval_fn` support
  - Auto-discovery of `GradComponent`.
  - Added `BackwardPassSetup` configuation

#### Evaluation
- Updated `RetrieverEvaluator` (formerly `RetrieverRecall`) with precision/recall
- Enhanced `AnswerMatchAcc` with:
  - `ExactMatch` from HotpotQA
  - `F1_score` implementation

#### Tooling & Utilities
- Revamped `tool_manager` with:
  - `CallFunctionTool` component
  - `FunctionExpressionToFunction` converter
- Updated `DataLoader` with seed support
- Improved `Prompt` system:
  - Converted to dataclass
  - Added list-to-string conversion in `prompt_kwargs`

## [0.2.7] - 2025-01-16

### Added
- `Memory` is completed with `call` and `add_dialog_turn` methods.
- Integrated `LanceDB` in the `Retriever`
- Multi-modal (image input and generation) in `OpenAIClient` along with tests.
- `ComponentList` to support a list of components registered in a component. Added `test_componentlist` to test the `ComponentList`.

### Improved
- Better diagnose report for `Trainer.diagnose`.
- `BedrockAPIClient` added more details on setup, yet it is still in experimental stage.
- `AzureAPIClient` added more details on setup, yet it is still in experimental stage.
-  `Retriever` class:
  - Support data id (field).
- `GradComponent`: Support pass-through gradient for the `forward` method.

Optimization
- Aggregated all backward engine prompts in `backward_engine_prompt`.
- Added `TGDData` for the optimizer to support reasoning at proposing new prompt.
- Added `sequential_order` in the `Trainer` to support the sequential training order. Reorganized the trainer code.
## [0.2.6] - 2024-11-25
### Improved
- Add default `max_tokens=512` to the `AnthropicAPIClient` to avoid the error when the user does not provide the `max_tokens` in the prompt.

## [0.2.5] - 2024-10-28

### Fixed
- `DataClassParser` nested data class parsing where we have to use `from_dict(json_dict)` instead of `(**json_dict)` to parse the nested data class.

## [0.2.4] - 2024-10-27

### Added
- `BedrockAPIClient` to support the AWS Bedrock API. Note: still need more testing and documentation.

### Fixed
- `Generator` cache path: fixed the issue with invalid window path with incompaticle special characters: https://github.com/SylphAI-Inc/AdalFlow/issues/237
- Wrong score in the diagnose file as `prepare_eval` is not passing classification label 0 to the `eval_input`.


## [0.2.3] - 2024-09-20

### Rename
- three methods within `AdalComponent` are renamed:
  - `handle_one_loss_sample` to `prepre_loss`
  - `handle_one_task_sample` to `prepre_task`
  - `evaluate_one_sample` to `prepre_eval`

## [0.2.3.beta.1] - 2024-09-17
### Removed
- Removed /reasoning as COT is just too simple to be a separate module.
### Fixed
- datasets/hotpotqa.py
- eval/answer_match_acc: added lower() to both the gt and pred in the fuzzy match. On hotpotqa, the accuracy goes from 0.15 to 0.4 on one test.
- eval/functional: fixed the `confidence_interval` to be able to customize the confidence level.

### Added
Auto-grad system to support retriever and any component:
- `GradComponent` has a default `forward` which wraps the `call` to handle the auto-grad automatically for any component that has subclassed `GradComponent`.
- Clarified the `ParamType` to include `input`, `output`, `hyperparam` instead of following PyTorch's tensor and Parameter design pattern.
- `TraceGraph` of the `Parameter` at `draw_graph` to support `ParamType`.
## [0.2.2] - 2024-09-09
### Added
- `get_cache_path`, instead of print out the cache path all the time, we add a ``get_cache_path`` to get the cache path.
- Make `huggingface datasets` as an optional dependency.
- Eval: `G_eval` to evaluate llm applications that have no reference text.
### Modified
- Add `template` to let users pass their own template, but need to have the same arguments as the default template.
- Added `checkpoint resume` in the `Trainer.diagnose` to show the newest performance and diagnostics on the checkpoint.

## [0.2.0] - 2024-08-20
### Added
- Qdrant retriever.

### Improved
- Add "mixed" training in ``Trainer`` to do demo and text optimization both in each step.
- ``DemoOptimizer``, allow to config if the input fields are included or excluded in the demonstration.
-  Added ``sequential`` and ``mix`` in the ``optimization_order`` in the ``Trainer`` to support the mixed training.
-  Added ``resume_from_ckpt`` in the ``Trainer.fit``.

### Fixed Bug
- wrong import in ``react`` agent.
## [0.2.0.beta.3] - 2024-08-16
### Fixed
- missing `diskcache` package in the dependencies.

## [0.2.0.beta.2] - 2024-08-15
### Improved
- ``Demo Optimizer`` with "learn-to-reason" one shot to achieve 94% on the object count, close to the 98% by the teacher gpt-4o model.

## [0.2.0.beta.1] - 2024-08-14
### Added
- Optimizer: `paramter`, `GradComponent`, `Optimizer`, `AdalComponent`, and `Trainer`.

### Added features
1.  ``DataClass``
* support ``__input_fields__``, ``get_input_fields``, ``set_input_fields__``, along with the output version.
* Support ``Parameter`` field which will be converted to `p.data``.
* Support ``include`` besides of ``exclude``, they cant coexist.
* ``to_dict`` made sure the ordering of fields is rearranged according to the ``input_fields`` and ``output_fields``.

## [0.1.0-beta.6] - 2024-07-23

Rename the `lightrag` package to `adalflow`.

### Fixed

- We introduced a bug in the previous release, the ollama model client non-streaming call was not working. We fixed it as we cannot put yield and return in the same function.

## [0.1.0-beta.5] - 2024-07-23

### Fixed
- [issue 134](https://github.com/SylphAI-Inc/AdalFlow/issues/134) Suppport Enum in `DataClass` schema. https://github.com/SylphAI-Inc/LightRAG/pull/135
- [issue 154](https://github.com/SylphAI-Inc/AdalFlow/issues/154) Fixed the `DataClass.from_dict` failure on `list[int]` type due to conditional check failure in the functional.

### Added
- Support streaming in Generator (sync call) [issue 149](https://github.com/SylphAI-Inc/AdalFlow/issues/149)
- Support streaming in OpenAIClient (sync call)

## [0.1.0-beta.3, 4] - 2024-07-18

### Added
- Ollama model client v1.

### Fixed
- func_tool.execute fails to run sync function in jupyter notebook. We avoid calling execute and instead in ToolManager, we
  add sync and async for each function method and call the `call` and `acall` method of func_tool directly.
- ModelClient acall bug.

### Improved
- Add function `extract_function_expression` in `functional`, and it will add missing right parenthesis to the function expression if LLMs fail to do so. This will make function expression more robust.

## [0.1.0-beta.2] - 2024-07-15

### Modified
- Make `LocalDB` a component for better visualization.
- Add extra packages in dependencides.

## [0.1.0-beta.1] - 2024-07-15

### Added
- `Sequential` adds `acall` method.
- Add extra packages so that users can install them with `pip install adalflow[extra]`.

## [0.0.0-beta.1] - 2024-07-10

### Added
- `DataClass`: add `__type_var_map__` in `data class schema` as the necessary step to support `Generic` in data class.
- Support Python `3.9`.

### Fixed
- `ReAct` agent is fixed to be working with updates on the json output parser.
- `Add` error handling for using Lazy Import classes the wrong way, such as subclass.



## [0.0.0-alpha.16] - 2024-07-08

### Fixed
- Anthropic client message does not use system role. For now, we put the whole prompt as the first user message.
- Update the `DEDEFAULT_LIGHTRAG_SYSTEM_PROMPT` to include 'You are a helpful assistant' as default <SYS> prompt.

## [0.0.0-alpha.15] - 2024-07-07

### Fixed
- `JsonOutputParser` and `YamlOutputParser` (1) forget to return the `output_dict` in the call. (2) Improve the output format str to ask it not to mistaken the "properties" and "type" as keys in the json/yaml output.
- Add test cases for the above fixes.

## [0.0.0-alpha.14] - 2024-07-06

### Modified
- `Sequential` moved from `lightrag.core.component` to `lightrag.core.container`.
- `Sequential` now accepts any positional and keyword arguments at the call time (the first component).
- `get_logger` in `lightrag.utils.logger` is simplified to config the root and named logger. Removed `enable_library_logging`.

### Fixed
- safe_import in `lightrag.components.model_client` to report import errors.
## [0.0.0-alpha.13] - 2024-07-04

### Added
- Added `return_data_class` parameter to `JsonOutputParser` and `YamlOutputParser` to return the data class object instead of the dictionary.

### Fixed
- Google client safe import.
- Postgres retriever compilation error.
- Database dialogTurn model.
