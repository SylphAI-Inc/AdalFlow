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
