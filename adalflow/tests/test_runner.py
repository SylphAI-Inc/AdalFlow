import unittest
import unittest.mock
import asyncio

from types import SimpleNamespace
from typing import List
from pydantic import BaseModel

from adalflow.core.runner import Runner
import adalflow.core.runner as runner_module
from adalflow.core.types import GeneratorOutput


class DummyFunction:
    """Mimics adalflow.core.types.Function."""
    def __init__(self, name, kwargs=None):
        self.name = name
        self.kwargs = kwargs or {}


class FakePlanner:
    """Planner stub that returns a sequence of GeneratorOutput or raw."""
    def __init__(self, outputs):
        self._outputs = outputs
        self._idx = 0

    def call(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
        out = self._outputs[self._idx]
        self._idx += 1
        return out

    async def acall(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
        return self.call(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
            use_cache=use_cache,
            id=id,
        )
    
    def get_prompt(self, **kwargs):
        return ""


class DummyAgent:
    """Bare-bones Agent for Runner, including answer_data_type for Runner.__init__."""
    def __init__(self, planner, max_steps=10, tool_manager=None, answer_data_type=None):
        self.planner = planner
        self.max_steps = max_steps
        self.tool_manager = tool_manager
        self.answer_data_type = answer_data_type


class DummyStepOutput:
    """Stub for StepOutput with flexible constructor."""
    def __init__(self, *args, **kwargs):
        self.step = kwargs.get('step', args[0] if len(args) > 0 else None)
        self.function = kwargs.get('function', args[1] if len(args) > 1 else None)
        # Runner uses 'observation'; fallback to 'output'
        self.output = kwargs.get('observation', kwargs.get('output', None))


class TestRunner(unittest.TestCase):
    def setUp(self):
        # Patch out the real StepOutput to avoid signature mismatch
        self.step_output_patcher = unittest.mock.patch.object(
            runner_module, 'StepOutput', DummyStepOutput
        )
        self.step_output_patcher.start()

        # Prepare a Runner with dummy agent
        self.runner = Runner(agent=DummyAgent(planner=None, answer_data_type=None))
        self.runner.stream_parser = None

    def tearDown(self):
        self.step_output_patcher.stop()

    def test_check_last_step(self):
        finish_fn = DummyFunction(name="finish")
        cont_fn = DummyFunction(name="continue")
        self.assertTrue(self.runner._check_last_step(finish_fn))
        self.assertFalse(self.runner._check_last_step(cont_fn))

    def test_call_single_step_finish(self):
        fn = DummyFunction(name="finish")
        go = GeneratorOutput(data=fn)
        agent = DummyAgent(planner=FakePlanner([go]), answer_data_type=None)
        runner = Runner(agent=agent)
        runner._tool_execute = lambda func: SimpleNamespace(output="done")

        history, last = runner.call(prompt_kwargs={})
        self.assertEqual(len(history), 1)
        self.assertIs(history[0].function, fn)
        self.assertEqual(last, "done")
        self.assertEqual(runner.step_history, history)

    def test_call_nonfinish_then_finish(self):
        fn1 = DummyFunction(name="step1")
        fn2 = DummyFunction(name="finish")
        go1, go2 = GeneratorOutput(data=fn1), GeneratorOutput(data=fn2)
        agent = DummyAgent(planner=FakePlanner([go1, go2]), answer_data_type=None)
        runner = Runner(agent=agent)
        runner._tool_execute = lambda func: SimpleNamespace(output=f"{func.name}_out")

        history, last = runner.call(prompt_kwargs={})
        names = [step.function.name for step in history]
        self.assertEqual(names, ["step1", "finish"])
        self.assertEqual(last, "finish_out")

    def test_call_respects_max_steps_without_finish(self):
        fn = DummyFunction(name="no_finish")
        go = GeneratorOutput(data=fn)
        agent = DummyAgent(planner=FakePlanner([go, go, go]), max_steps=2, answer_data_type=None)
        runner = Runner(agent=agent)
        runner._tool_execute = lambda func: SimpleNamespace(output="out")

        history, last = runner.call(prompt_kwargs={})
        self.assertEqual(len(history), 2)
        self.assertEqual([s.function.name for s in history], ["no_finish", "no_finish"])
        self.assertEqual(last, "out")

    def test_call_invalid_answer_data_type(self):
        agent = DummyAgent(planner=FakePlanner([None]), answer_data_type=None)
        runner = Runner(agent=agent)
        runner._tool_execute = lambda func: SimpleNamespace(output="x")

        result = runner.call(prompt_kwargs={})
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Error in step 0:"))

    def test_acall_single_step(self):
        fn = DummyFunction(name="finish")
        go = GeneratorOutput(data=fn)
        agent = DummyAgent(planner=FakePlanner([go]), answer_data_type=None)
        runner = Runner(agent=agent)
        runner._tool_execute = lambda func: SimpleNamespace(output="async-done")

        history, last = asyncio.run(runner.acall(prompt_kwargs={}))
        self.assertEqual(len(history), 1)
        self.assertEqual(last, "async-done")
        self.assertEqual(runner.step_history, history)

    def test_acall_invalid_answer_data_type(self):
        agent = DummyAgent(planner=FakePlanner([None]), answer_data_type=None)
        runner = Runner(agent=agent)
        runner._tool_execute = lambda func: SimpleNamespace(output="x")

        history, err = asyncio.run(runner.acall(prompt_kwargs={}))
        self.assertIsInstance(err, str)
        self.assertTrue(err.startswith("Error in step 0:"))
        self.assertEqual(history, [])

    def test_process_data_without_answer_data_type(self):
        out = self.runner._process_data(data="raw", id=None)
        self.assertEqual(out, "raw")

    def test_process_data_with_valid_pydantic_model(self):
        class M(BaseModel):
            a: int
            b: str

        runner = Runner(agent=DummyAgent(planner=None, answer_data_type=M))
        data = {"properties": {"a": 5, "b": "ok"}}  # Wrap in properties
        result = runner._process_data(data)
        self.assertIsInstance(result, M)
        self.assertEqual(result.a, 5)
        self.assertEqual(result.b, "ok")

    def test_process_data_with_nested_objects(self):
        class Nested(BaseModel):
            value: str
            count: int

        class M(BaseModel):
            name: str
            nested: Nested

        runner = Runner(agent=DummyAgent(planner=None, answer_data_type=M))
        data = {
            "properties": {
                "name": "test",
                "nested": {
                    "properties": {
                        "value": "hello",
                        "count": 42
                    }
                }
            }
        }
        result = runner._process_data(data)
        self.assertIsInstance(result, M)
        self.assertIsInstance(result.nested, Nested)
        self.assertEqual(result.name, "test")
        self.assertEqual(result.nested.value, "hello")
        self.assertEqual(result.nested.count, 42)

    def test_process_data_with_list_of_objects(self):
        class Item(BaseModel):
            id: int
            name: str

        class M(BaseModel):
            items: List[Item]

        runner = Runner(agent=DummyAgent(planner=None, answer_data_type=M))
        data = {
            "properties": {
                "items": [
                    {"properties": {"id": 1, "name": "one"}},
                    {"properties": {"id": 2, "name": "two"}}
                ]
            }
        }
        result = runner._process_data(data)
        self.assertIsInstance(result, M)
        self.assertEqual(len(result.items), 2)
        self.assertIsInstance(result.items[0], Item)
        self.assertEqual(result.items[0].id, 1)
        self.assertEqual(result.items[1].name, "two")


if __name__ == "__main__":
    unittest.main()
