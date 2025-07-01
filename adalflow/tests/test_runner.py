# import unittest
# import unittest.mock
# import asyncio
# import json
# from dataclasses import dataclass
# from datetime import datetime, timezone
# from types import SimpleNamespace
# from typing import List, Dict, Any
# from pydantic import BaseModel, Field
# from adalflow.core.types import GeneratorOutput, Function

# from adalflow.core.runner import Runner
# import adalflow.core.runner as runner_module


# class DummyFunction(Function):
#     """Mimics adalflow.core.types.Function."""

#     def __init__(self, name, kwargs=None):
#         super().__init__(name=name, args=kwargs or {})


# class FakePlanner:
#     """Planner stub that returns a sequence of GeneratorOutput or raw."""

#     def __init__(self, outputs):
#         # Wrap outputs in GeneratorOutput if they're not already
#         self._outputs = [
#             out if isinstance(out, GeneratorOutput) else GeneratorOutput(data=out)
#             for out in outputs
#         ]
#         self._idx = 0

#     def call(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
#         if self._idx >= len(self._outputs):
#             raise IndexError("No more outputs")
#         out = self._outputs[self._idx]
#         self._idx += 1
#         return out

#     async def acall(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
#         return self.call(
#             prompt_kwargs=prompt_kwargs,
#             model_kwargs=model_kwargs,
#             use_cache=use_cache,
#             id=id,
#         )

#     def get_prompt(self, **kwargs):
#         return ""


# class DummyAgent:
#     """Bare-bones Agent for Runner, including answer_data_type for Runner.__init__."""

#     def __init__(self, planner, max_steps=10, tool_manager=None, answer_data_type=None):
#         self.planner = planner
#         self.max_steps = max_steps
#         self.tool_manager = tool_manager
#         self.answer_data_type = answer_data_type


# class DummyStepOutput:
#     """Stub for StepOutput with flexible constructor."""

#     def __init__(self, *args, **kwargs):
#         self.step = kwargs.get("step", args[0] if len(args) > 0 else None)
#         self.function = kwargs.get("function", args[1] if len(args) > 1 else None)
#         # Runner uses 'observation'; fallback to 'output'
#         self.observation = kwargs.get("observation", kwargs.get("output", None))


# # Test Models
# class NestedModel(BaseModel):
#     id: int
#     name: str
#     created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# class UserModel(BaseModel):
#     id: int
#     username: str
#     email: str
#     profile: Dict[str, Any]
#     nested: NestedModel
#     tags: List[str] = []


# @dataclass
# class RegularDataclass:
#     id: int
#     name: str
#     active: bool = True


# class ComplexResponse(BaseModel):
#     users: List[UserModel]
#     timestamp: datetime
#     metadata: Dict[str, Any]
#     status: str
#     score: float


# class TestRunner(unittest.TestCase):
#     def setUp(self):
#         # Patch out the real StepOutput to avoid signature mismatch
#         self.step_output_patcher = unittest.mock.patch.object(
#             runner_module, "StepOutput", DummyStepOutput
#         )
#         self.step_output_patcher.start()

#         # Prepare a Runner with dummy agent
#         self.runner = Runner(agent=DummyAgent(planner=None, answer_data_type=None))
#         self.runner.stream_parser = None

#         # Sample test data
#         self.sample_nested = NestedModel(id=1, name="test")
#         self.sample_user = UserModel(
#             id=1,
#             username="testuser",
#             email="test@example.com",
#             profile={"role": "admin", "permissions": ["read", "write"]},
#             nested=self.sample_nested,
#             tags=["active", "verified"],
#         )
#         self.sample_dataclass = RegularDataclass(id=1, name="test")
#         self.sample_complex = ComplexResponse(
#             users=[self.sample_user],
#             timestamp=datetime.now(timezone.utc),
#             metadata={"page": 1, "total": 100},
#             status="success",
#             score=99.5,
#         )

#     def tearDown(self):
#         self.step_output_patcher.stop()

#     def test_check_last_step(self):
#         finish_fn = DummyFunction(name="finish")
#         cont_fn = DummyFunction(name="continue")
#         self.assertTrue(self.runner._check_last_step(finish_fn))
#         self.assertFalse(self.runner._check_last_step(cont_fn))

#     def test_call_single_step_finish(self):
#         fn = DummyFunction(name="finish")
#         # Create a mock tool manager that returns a FunctionOutput
#         mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output="done")
#         agent = DummyAgent(
#             planner=FakePlanner([GeneratorOutput(data=fn)]),
#             answer_data_type=None,
#             tool_manager=mock_tool_manager
#         )
#         runner = Runner(agent=agent)

#         history, result = runner.call(prompt_kwargs={})
#         self.assertEqual(len(history), 1)
#         self.assertIs(history[0].function, fn)
#         self.assertEqual(result, "done")
#         self.assertEqual(runner.step_history, history)

#     def test_call_nonfinish_then_finish(self):
#         fn1 = DummyFunction(name="search")
#         fn2 = DummyFunction(name="finish")
#         # Create a mock tool manager that returns a FunctionOutput
#         mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output="test output")
#         agent = DummyAgent(
#             planner=FakePlanner([GeneratorOutput(data=fn1), GeneratorOutput(data=fn2)]),
#             answer_data_type=None,
#             tool_manager=mock_tool_manager
#         )
#         runner = Runner(agent=agent)

#         history, result = runner.call(prompt_kwargs={})
#         self.assertEqual(len(history), 2)
#         self.assertEqual(result, "test output")

#     def test_call_respects_max_steps_without_finish(self):
#         # Create outputs for 5 steps without finish
#         functions = [DummyFunction(name=f"action_{i}") for i in range(5)]
#         outputs = [GeneratorOutput(data=fn) for fn in functions]
#         # Create a mock tool manager that returns a FunctionOutput
#         mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output=f"output_{expr_or_fun.name}")
#         agent = DummyAgent(
#             planner=FakePlanner(outputs),
#             max_steps=3,
#             answer_data_type=None,
#             tool_manager=mock_tool_manager
#         )
#         runner = Runner(agent=agent)

#         history, result = runner.call(prompt_kwargs={})
#         # Should only execute 3 steps due to max_steps limit
#         self.assertEqual(len(history), 3)
#         self.assertIsNone(result)  # No finish step was reached
#         # Check that the correct functions were executed
#         for i, step in enumerate(history):
#             self.assertEqual(step.function.name, f"action_{i}")
#             self.assertEqual(step.observation, f"output_action_{i}")

#     def test_call_no_answer_data_type(self):
#         fn = DummyFunction(name="finish")
#         # Create a mock tool manager that returns a FunctionOutput
#         mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output={"result": "success"})
#         agent = DummyAgent(
#             planner=FakePlanner([GeneratorOutput(data=fn)]),
#             answer_data_type=None,
#             tool_manager=mock_tool_manager
#         )
#         runner = Runner(agent=agent)

#         history, result = runner.call(prompt_kwargs={})
#         self.assertEqual(len(history), 1)
#         self.assertEqual(result, {"result": "success"})

#     def test_acall_single_step(self):
#         async def async_test():
#             from adalflow.core.types import FunctionOutput
#             fn = DummyFunction(name="finish")
#             agent = DummyAgent(
#                 planner=FakePlanner([GeneratorOutput(data=fn)]), answer_data_type=None
#             )
#             runner = Runner(agent=agent)

#             async def mock_tool_execute_async(func):
#                 return FunctionOutput(name=func.name, input=func, output="async-done")

#             runner._tool_execute_async = mock_tool_execute_async

#             history, result = await runner.acall(prompt_kwargs={})
#             self.assertEqual(len(history), 1)
#             self.assertEqual(result, "async-done")
#             self.assertEqual(runner.step_history, history)

#         asyncio.run(async_test())

#     def test_acall_invalid_answer_data_type(self):
#         agent = DummyAgent(planner=FakePlanner([None]), answer_data_type=None)
#         runner = Runner(agent=agent)
#         runner._tool_execute = lambda func: SimpleNamespace(output="x")

#         history, err = asyncio.run(runner.acall(prompt_kwargs={}))
#         self.assertIsInstance(err, str)
#         self.assertTrue(err.startswith("Error in step 0:"))
#         self.assertEqual(history, [])

#     def test_process_data_without_answer_data_type(self):
#         out = self.runner._process_data(data="raw", id=None)
#         self.assertEqual(out, "raw")

#     def test_tool_execute_sync_with_sync_function(self):
#         """Test _tool_execute_sync with synchronous function result."""
#         mock_function = DummyFunction(name="test_func")
#         mock_result = SimpleNamespace(output="sync_result")

#         # Mock the tool_manager to return a sync result
#         self.runner.agent.tool_manager = unittest.mock.Mock(return_value=mock_result)

#         result = self.runner._tool_execute_sync(mock_function)
#         self.assertEqual(result, mock_result)

#     def test_tool_execute_sync_with_async_function_no_loop(self):
#         """Test _tool_execute_sync with async function when no event loop is running."""
#         import asyncio

#         mock_function = DummyFunction(name="test_async_func")
#         mock_result = SimpleNamespace(output="async_result")

#         async def async_mock():
#             return mock_result

#         # Mock the tool_manager to return a coroutine
#         self.runner.agent.tool_manager = unittest.mock.Mock(return_value=async_mock())

#         result = self.runner._tool_execute_sync(mock_function)
#         self.assertEqual(result, mock_result)

#     def test_tool_execute_sync_with_async_generator_no_loop(self):
#         """Test _tool_execute_sync with async generator when no event loop is running."""
#         import asyncio

#         mock_function = DummyFunction(name="test_async_gen")

#         async def async_generator():
#             yield SimpleNamespace(output="item1")
#             yield SimpleNamespace(output="item2")
#             yield SimpleNamespace(output="final_result")

#         # Mock the tool_manager to return an async generator
#         self.runner.agent.tool_manager = unittest.mock.Mock(return_value=async_generator())

#         result = self.runner._tool_execute_sync(mock_function)
#         self.assertEqual(result.output, "final_result")

#     # temporary disable the test for process_data

#     # def test_process_data_with_valid_pydantic_model(self):
#     #     class M(BaseModel):
#     #         a: int
#     #         b: str

#     #     runner = Runner(agent=DummyAgent(planner=None, answer_data_type=M))
#     #     data = {"properties": {"a": 5, "b": "ok"}}  # Wrap in properties
#     #     result = runner._process_data(data)
#     #     self.assertIsInstance(result, M)
#     #     self.assertEqual(result.a, 5)
#     #     self.assertEqual(result.b, "ok")

#     # def test_process_data_with_nested_objects(self):
#     #     class Nested(BaseModel):
#     #         value: str
#     #         count: int

#     #     class M(BaseModel):
#     #         name: str
#     #         nested: Nested

#     #     runner = Runner(agent=DummyAgent(planner=None, answer_data_type=M))
#     #     data = {
#     #         "properties": {
#     #             "name": "test",
#     #             "nested": {"properties": {"value": "hello", "count": 42}},
#     #         }
#     #     }
#     #     result = runner._process_data(data)
#     #     self.assertIsInstance(result, M)
#     #     self.assertIsInstance(result.nested, Nested)
#     #     self.assertEqual(result.name, "test")
#     #     self.assertEqual(result.nested.value, "hello")
#     #     self.assertEqual(result.nested.count, 42)

#     # def test_process_data_with_list_of_objects(self):
#     #     class Item(BaseModel):
#     #         id: int
#     #         name: str

#     #     class M(BaseModel):
#     #         items: List[Item]

#     #     runner = Runner(agent=DummyAgent(planner=None, answer_data_type=M))
#     #     data = {
#     #         "properties": {
#     #             "items": [
#     #                 {"properties": {"id": 1, "name": "one"}},
#     #                 {"properties": {"id": 2, "name": "two"}},
#     #             ]
#     #         }
#     #     }
#     #     result = runner._process_data(data)
#     #     self.assertIsInstance(result, M)
#     #     self.assertEqual(len(result.items), 2)
#     #     self.assertIsInstance(result.items[0], Item)
#     #     self.assertEqual(result.items[0].id, 1)
#     #     self.assertEqual(result.items[1].name, "two")

#     def test_process_data_with_complex_nested_models(self):
#         runner = Runner(agent=DummyAgent(planner=None, answer_data_type=UserModel))
#         data = {
#             "id": 1,
#             "username": "testuser",
#             "email": "test@example.com",
#             "profile": {"role": "admin", "permissions": ["read", "write"]},
#             "nested": {"id": 1, "name": "test"},
#             "tags": ["active", "verified"],
#         }
#         result = runner._process_data(json.dumps(data))
#         self.assertIsInstance(result, UserModel)
#         self.assertEqual(result.username, "testuser")
#         self.assertIsInstance(result.nested, NestedModel)
#         self.assertEqual(result.nested.name, "test")
#         self.assertIn("admin", result.profile["role"])

#     def test_acall_with_complex_structure(self):
#         async def async_test():
#             from adalflow.core.types import FunctionOutput
#             # Create a finish function with the complex data
#             finish_fn = DummyFunction(
#                 name="finish", kwargs={"output": json.dumps(self.sample_complex.model_dump())}
#             )
#             agent = DummyAgent(
#                 planner=FakePlanner([GeneratorOutput(data=finish_fn)]),
#                 answer_data_type=ComplexResponse,
#             )
#             runner = Runner(agent=agent)

#             async def mock_tool_execute_async(func):
#                 return FunctionOutput(
#                     name=func.name,
#                     input=func,
#                     output=func.kwargs.get("output")
#                 )

#             runner._tool_execute_async = mock_tool_execute_async

#             # Test async call
#             history, result = await runner.acall(prompt_kwargs={})
#             self.assertIsInstance(result, ComplexResponse)
#             self.assertEqual(result.status, "success")
#             self.assertIsInstance(result.users[0], UserModel)
#             self.assertEqual(result.users[0].username, "testuser")

#         asyncio.run(async_test())

#     def test_call_with_nested_structures(self):
#         # Mock the agent's planner to return our test data
#         mock_planner = FakePlanner(
#             [
#                 GeneratorOutput(
#                     data=DummyFunction(
#                         name="finish",
#                         kwargs={
#                             "output": json.dumps(
#                                 {
#                                     "id": 1,
#                                     "username": "testuser",
#                                     "email": "test@example.com",
#                                     "profile": {
#                                         "role": "admin",
#                                         "permissions": ["read", "write"],
#                                     },
#                                     "nested": {"id": 1, "name": "test"},
#                                     "tags": ["active", "verified"],
#                                 }
#                             )
#                         },
#                     )
#                 )
#             ]
#         )
#         agent = DummyAgent(planner=mock_planner, answer_data_type=UserModel)
#         runner = Runner(agent=agent)

#         # Test sync call
#         with self.assertRaises(ValueError):
#             history, result = runner.call(prompt_kwargs={})
#             self.assertIsInstance(result, UserModel)
#             self.assertEqual(result.username, "testuser")
#             self.assertIsInstance(result.nested, NestedModel)
#             self.assertIn("active", result.tags)

#     def test_process_data_with_datetime_parsing(self):
#         runner = Runner(agent=DummyAgent(planner=None, answer_data_type=NestedModel))
#         test_time = datetime.now(timezone.utc).isoformat()
#         data = f'{{"id": 1, "name": "test", "created_at": "{test_time}"}}'
#         result = runner._process_data(data)
#         self.assertIsInstance(result, NestedModel)
#         self.assertEqual(result.id, 1)
#         self.assertEqual(result.name, "test")

#     def test_process_data_with_invalid_data(self):
#         # Create test data with missing required 'name' field
#         test_time = datetime.now(timezone.utc).isoformat()
#         invalid_data = '{"id": 1, "created_at": "' + test_time + '"}'  # Missing 'name'

#         # Create a mock planner that will return invalid data
#         mock_planner = FakePlanner(
#             [
#                 GeneratorOutput(
#                     data=DummyFunction(name="finish", kwargs={"output": invalid_data})
#                 )
#             ]
#         )
#         agent = DummyAgent(planner=mock_planner, answer_data_type=NestedModel)
#         runner = Runner(agent=agent)
#         runner._tool_execute = lambda func: SimpleNamespace(
#             output=func.kwargs.get("output")
#         )

#         # The error should be raised during processing
#         with self.assertRaises(ValueError) as ctx:
#             runner.call(prompt_kwargs={})
#         self.assertIn("Error in step 0:", str(ctx.exception))

#     def test_process_data_with_list_of_objects(self):
#         class Item(BaseModel):
#             id: int
#             name: str

#     #     class M(BaseModel):
#     #         items: List[Item]

#     #     runner = Runner(agent=DummyAgent(planner=None, answer_data_type=M))
#     #     data = {
#     #         "properties": {
#     #             "items": [
#     #                 {"properties": {"id": 1, "name": "one"}},
#     #                 {"properties": {"id": 2, "name": "two"}},
#     #             ]
#     #         }
#     #     }
#     #     result = runner._process_data(data)
#     #     self.assertIsInstance(result, M)
#     #     self.assertEqual(len(result.items), 2)
#     #     self.assertIsInstance(result.items[0], Item)
#     #     self.assertEqual(result.items[0].id, 1)
#     #     self.assertEqual(result.items[1].name, "two")


# if __name__ == "__main__":
#     unittest.main()
