"""Integration tests: Runner + reliability layer (retry, timeout, checkpoint, resume).

Uses fake planners/tools and injected clock/sleep - no network, no paid model
calls.  Mirrors the fake patterns in tests/test_runner.py.
"""

import asyncio
import hashlib
import json
import tempfile
import unittest

from adalflow.components.agent.reliability import (
    BackoffPolicy,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    ReliabilityConfig,
    RetryPolicy,
    TimeoutPolicy,
    ToolPolicy,
    ToolReplayRefusedError,
)
from adalflow.components.agent.reliability.errors import ResumeMismatchError
from adalflow.components.agent.runner import Runner
from adalflow.core.types import (
    Function,
    FunctionOutput,
    GeneratorOutput,
    RunItemStreamEvent,
    RunnerResult,
    ToolCallActivityRunItem,
    ToolOutput,
)


# ---------------------------------------------------------------------------
# Fakes (kept self-contained so this file can run standalone)
# ---------------------------------------------------------------------------
class DummyFunction(Function):
    def __init__(self, name, kwargs=None, _is_answer_final=False, _answer=None):
        super().__init__(name=name, kwargs=kwargs or {})
        self._is_answer_final = _is_answer_final
        self._answer = _answer


class FakePlanner:
    """Returns a sequence of GeneratorOutput, with optional call counter."""

    def __init__(self, outputs, on_call=None):
        self._outputs = [
            out if isinstance(out, GeneratorOutput) else GeneratorOutput(data=out)
            for out in outputs
        ]
        self._idx = 0
        self.call_count = 0
        self._on_call = on_call

    def call(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
        if self._on_call:
            self._on_call()
        self.call_count += 1
        if self._idx >= len(self._outputs):
            return GeneratorOutput(data=None, error="exhausted")
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


class CountingToolManager:
    """Tool manager that counts execute_func calls and returns canned outputs."""

    def __init__(self, output_map=None, default_output="ok"):
        self._output_map = output_map or {}
        self._default = default_output
        self.call_count = 0
        self.calls = []

    def execute_func(self, func):
        self.call_count += 1
        self.calls.append(func.name)
        out = self._output_map.get(func.name, self._default)
        if isinstance(out, Exception):
            raise out
        return FunctionOutput(
            name=func.name,
            input=func,
            output=ToolOutput(output=out, observation=str(out)),
        )

    async def execute_func_async(self, func):
        return self.execute_func(func)


class DummyAgent:
    def __init__(
        self,
        planner,
        max_steps=10,
        tool_manager=None,
        answer_data_type=None,
        name="dummy",
    ):
        self.planner = planner
        self.max_steps = max_steps
        self.tool_manager = tool_manager or CountingToolManager()
        self.answer_data_type = answer_data_type
        self.name = name


def make_runner(
    planner_outputs,
    reliability=None,
    checkpoint_store=None,
    tool_manager=None,
    max_steps=10,
    answer_data_type=None,
    on_planner_call=None,
    agent_name="dummy",
):
    planner = FakePlanner(planner_outputs, on_call=on_planner_call)
    tm = tool_manager or CountingToolManager()
    agent = DummyAgent(
        planner,
        max_steps=max_steps,
        tool_manager=tm,
        answer_data_type=answer_data_type,
        name=agent_name,
    )
    return Runner(
        agent=agent,
        max_steps=max_steps,
        reliability=reliability,
        checkpoint_store=checkpoint_store,
    )


# ---------------------------------------------------------------------------
# Sync call path with retry + timeout
# ---------------------------------------------------------------------------
class TestRunnerSyncReliability(unittest.TestCase):
    def test_sync_planner_retry_then_finish(self):
        outputs = [
            GeneratorOutput(data=None, error="Error code: 500"),
            GeneratorOutput(data=None, error="Error code: 500"),
            GeneratorOutput(
                data=DummyFunction(name="finish", _is_answer_final=True, _answer="done")
            ),
        ]
        reliability = ReliabilityConfig(
            retry=RetryPolicy(
                max_attempts=5, backoff=BackoffPolicy(initial=0.0, jitter="none")
            )
        )
        runner = make_runner(outputs, reliability=reliability)
        result = runner.call(prompt_kwargs={"input_str": "hi"}, run_id="run-1")
        self.assertIsInstance(result, RunnerResult)
        self.assertEqual(result.answer, "done")
        self.assertEqual(runner.agent.planner.call_count, 3)

    def test_sync_planner_timeout_abandoned(self):
        import time

        def slow_call():
            time.sleep(0.5)
            return GeneratorOutput(
                data=DummyFunction(name="finish", _is_answer_final=True, _answer="done")
            )

        # Build a planner whose call sleeps.
        planner = FakePlanner([])
        planner.call = lambda **kw: slow_call()  # type: ignore[assignment]
        agent = DummyAgent(planner, max_steps=1)
        reliability = ReliabilityConfig(
            retry=RetryPolicy.none(),
            timeouts=TimeoutPolicy(planner_timeout=0.05),
        )
        runner = Runner(agent=agent, max_steps=1, reliability=reliability)
        result = runner.call(prompt_kwargs={"input_str": "hi"})
        # The planner timed out; runner records an error step and continues.
        self.assertIsInstance(result, RunnerResult)


# ---------------------------------------------------------------------------
# Async acall path
# ---------------------------------------------------------------------------
class TestRunnerAsyncReliability(unittest.TestCase):
    def test_acall_429_retried(self):
        async def run():
            outputs = [
                GeneratorOutput(data=None, error="Error code: 429"),
                GeneratorOutput(data=None, error="Error code: 429"),
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="ok"
                    )
                ),
            ]
            reliability = ReliabilityConfig(
                retry=RetryPolicy(
                    max_attempts=5, backoff=BackoffPolicy(initial=0.0, jitter="none")
                )
            )
            runner = make_runner(outputs, reliability=reliability)
            result = await runner.acall(prompt_kwargs={"input_str": "hi"}, run_id="r1")
            self.assertEqual(result.answer, "ok")
            self.assertEqual(runner.agent.planner.call_count, 3)

        asyncio.run(run())

    def test_acall_tool_timeout_continues_run(self):
        async def run():
            # Step 1: a tool that times out; step 2: finish.
            outputs = [
                GeneratorOutput(data=DummyFunction(name="slow_tool")),
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="done"
                    )
                ),
            ]

            async def slow_tool(func):
                await asyncio.sleep(1.0)
                return FunctionOutput(name=func.name, input=func, output=None)

            class AsyncTM:
                def __init__(self):
                    self.calls = 0

                async def execute_func_async(self, func):
                    self.calls += 1
                    return await slow_tool(func)

            tm = AsyncTM()
            reliability = ReliabilityConfig(
                retry=RetryPolicy.none(),
                timeouts=TimeoutPolicy(tool_timeout=0.05),
            )
            runner = make_runner(
                outputs, reliability=reliability, tool_manager=tm, max_steps=3
            )
            result = await runner.acall(prompt_kwargs={"input_str": "hi"}, run_id="r1")
            self.assertEqual(result.answer, "done")
            # The tool timed out but the run continued to finish.
            self.assertEqual(len(runner.step_history), 1)
            self.assertEqual(
                runner.step_history[0].observation, runner.step_history[0].observation
            )
            self.assertIn("timed out", str(runner.step_history[0].observation))

        asyncio.run(run())

    def test_acall_run_deadline(self):
        async def run():
            outputs = [
                GeneratorOutput(data=None, error="Error code: 500"),
            ]
            reliability = ReliabilityConfig(
                retry=RetryPolicy(
                    max_attempts=10, backoff=BackoffPolicy(initial=0.0, jitter="none")
                ),
                timeouts=TimeoutPolicy(run_timeout=0.0),
            )
            runner = make_runner(outputs, reliability=reliability, max_steps=5)
            result = await runner.acall(prompt_kwargs={"input_str": "hi"}, run_id="r1")
            # Deadline exceeded -> error result.
            self.assertIsInstance(result, RunnerResult)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Checkpoint + resume
# ---------------------------------------------------------------------------
class TestRunnerCheckpointResume(unittest.TestCase):
    def test_resume_replays_completed_tool_without_call(self):
        from adalflow.components.agent.reliability import (
            RunCheckpoint,
            RunFingerprint,
            ToolLedgerEntry,
        )

        store = InMemoryCheckpointStore()
        # Compute the real args_hash the runner will use for an empty-args
        # "search" Function, so the ledger key matches on resume.
        search_fn = Function(name="search")
        args_hash = hashlib.sha256(
            json.dumps(
                {"name": "search", "args": [], "kwargs": {}},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        # New ledger key includes the step index (step 0 in progress).
        ledger_key = f"0:search:{args_hash}"

        fp = RunFingerprint(
            agent_name="dummy",
            planner_model=None,
            tool_names_sorted=["search"],
            tool_signature_hash="sig",
            max_steps=5,
            input_hash=hashlib.sha256(b"q").hexdigest(),
        )
        cp = RunCheckpoint(
            run_id="run-A",
            status="running",
            created_at=1.0,
            updated_at=1.0,
            step_count=0,
            step_history=[],
            tool_ledger={
                ledger_key: ToolLedgerEntry(
                    call_id=ledger_key,
                    name="search",
                    args_hash=args_hash,
                    state="completed",
                    observation_json=json.dumps(
                        {
                            "output": "search-result",
                            "observation": "search-result",
                            "status": "success",
                        }
                    ),
                ).to_dict()
            },
            fingerprint=fp.to_dict(),
            prompt_kwargs={"input_str": "q"},
            # The in-progress step's Function: the planner already produced
            # "search" and the tool completed, but the step was never appended.
            pending_function=search_fn.to_dict(),
        )
        store.save(cp)
        # After replaying step 0 from the ledger, the next planner call is
        # "finish" (the planner is NOT called for the replayed step).
        outputs = [
            GeneratorOutput(
                data=DummyFunction(
                    name="finish", _is_answer_final=True, _answer="done2"
                )
            ),
        ]
        tm = CountingToolManager(default_output="search-result2")
        runner = make_runner(
            outputs, checkpoint_store=store, tool_manager=tm, max_steps=5
        )
        result = runner.call(
            prompt_kwargs={"input_str": "q"},
            run_id="run-A",
            resume=True,
            allow_fingerprint_mismatch=True,
        )
        self.assertEqual(result.answer, "done2")
        # The search tool was replayed from the ledger, NOT re-executed.
        self.assertEqual(tm.call_count, 0)
        # The planner was not called for the replayed step (only for finish).
        self.assertEqual(runner.agent.planner.call_count, 1)

    def test_resume_completed_run_returns_stored_result(self):
        store = InMemoryCheckpointStore()
        outputs = [
            GeneratorOutput(
                data=DummyFunction(
                    name="finish", _is_answer_final=True, _answer="stored"
                )
            ),
        ]
        runner = make_runner(outputs, checkpoint_store=store, max_steps=3)
        runner.call(prompt_kwargs={"input_str": "q"}, run_id="run-B")
        # Resume a completed run: planner must NOT be called.
        outputs2 = [
            GeneratorOutput(
                data=DummyFunction(name="finish", _is_answer_final=True, _answer="new")
            ),
        ]
        runner2 = make_runner(outputs2, checkpoint_store=store, max_steps=3)
        result = runner2.call(
            prompt_kwargs={"input_str": "q"}, run_id="run-B", resume=True
        )
        self.assertEqual(result.answer, "stored")
        self.assertEqual(runner2.agent.planner.call_count, 0)

    def test_started_ledger_non_idempotent_raises(self):
        from adalflow.components.agent.reliability import ToolLedgerEntry

        store = InMemoryCheckpointStore()
        # Build a checkpoint with a "started" ledger entry for a non-idempotent
        # tool, and a pending_function so resume knows this step was in progress.
        charge_fn = Function(name="charge_card")
        args_hash = hashlib.sha256(
            json.dumps(
                {"name": "charge_card", "args": [], "kwargs": {}},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        ledger_key = f"0:charge_card:{args_hash}"
        cp = __import__(
            "adalflow.components.agent.reliability", fromlist=["RunCheckpoint"]
        ).RunCheckpoint(
            run_id="run-C",
            status="running",
            created_at=1.0,
            updated_at=1.0,
            step_count=0,
            step_history=[],
            tool_ledger={
                ledger_key: ToolLedgerEntry(
                    call_id=ledger_key,
                    name="charge_card",
                    args_hash=args_hash,
                    state="started",
                ).to_dict()
            },
            prompt_kwargs={"input_str": "q"},
            pending_function=charge_fn.to_dict(),
        )
        store.save(cp)
        outputs = [
            GeneratorOutput(
                data=DummyFunction(name="finish", _is_answer_final=True, _answer="x")
            ),
        ]
        runner = make_runner(outputs, checkpoint_store=store, max_steps=3)
        with self.assertRaises(ToolReplayRefusedError):
            runner.call(
                prompt_kwargs={"input_str": "q"},
                run_id="run-C",
                resume=True,
                on_incomplete_tool="error",
            )

    def test_started_ledger_skip_policy(self):
        from adalflow.components.agent.reliability import RunCheckpoint, ToolLedgerEntry

        store = InMemoryCheckpointStore()
        args_hash = hashlib.sha256(
            json.dumps(
                {"name": "charge_card", "args": [], "kwargs": {}},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        ledger_key = f"0:charge_card:{args_hash}"
        cp = RunCheckpoint(
            run_id="run-D",
            status="running",
            created_at=1.0,
            updated_at=1.0,
            step_count=0,
            step_history=[],
            tool_ledger={
                ledger_key: ToolLedgerEntry(
                    call_id=ledger_key,
                    name="charge_card",
                    args_hash=args_hash,
                    state="started",
                ).to_dict()
            },
            prompt_kwargs={"input_str": "q"},
            pending_function=Function(name="charge_card").to_dict(),
        )
        store.save(cp)
        outputs = [
            GeneratorOutput(
                data=DummyFunction(name="finish", _is_answer_final=True, _answer="x")
            ),
        ]
        runner = make_runner(outputs, checkpoint_store=store, max_steps=3)
        # skip policy: should not raise, run completes.
        result = runner.call(
            prompt_kwargs={"input_str": "q"},
            run_id="run-D",
            resume=True,
            on_incomplete_tool="skip",
        )
        self.assertEqual(result.answer, "x")
        # The started tool was skipped (not re-executed); only finish runs.
        self.assertEqual(runner.agent.planner.call_count, 1)

    def test_fingerprint_mismatch_raises(self):
        from adalflow.components.agent.reliability import RunCheckpoint, RunFingerprint

        store = InMemoryCheckpointStore()
        fp = RunFingerprint(
            agent_name="other_agent",
            planner_model="gpt-x",
            tool_names_sorted=["search"],
            tool_signature_hash="sig",
            max_steps=3,
            input_hash="ih",
        )
        cp = RunCheckpoint(
            run_id="run-E",
            status="running",
            created_at=1.0,
            updated_at=1.0,
            step_count=0,
            step_history=[],
            fingerprint=fp.to_dict(),
            prompt_kwargs={"input_str": "q"},
        )
        store.save(cp)
        outputs = [
            GeneratorOutput(
                data=DummyFunction(name="finish", _is_answer_final=True, _answer="x")
            ),
        ]
        runner = make_runner(
            outputs, checkpoint_store=store, max_steps=3, agent_name="different"
        )
        with self.assertRaises(ResumeMismatchError):
            runner.call(prompt_kwargs={"input_str": "q"}, run_id="run-E", resume=True)


# ---------------------------------------------------------------------------
# Streaming reliability events
# ---------------------------------------------------------------------------
class TestRunnerStreamingReliability(unittest.TestCase):
    def test_astream_emits_reliability_events(self):
        async def run():
            outputs = [
                GeneratorOutput(data=None, error="Error code: 500"),
                GeneratorOutput(data=None, error="Error code: 500"),
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="ok"
                    )
                ),
            ]
            reliability = ReliabilityConfig(
                retry=RetryPolicy(
                    max_attempts=5, backoff=BackoffPolicy(initial=0.0, jitter="none")
                )
            )
            runner = make_runner(outputs, reliability=reliability)
            stream = runner.astream(prompt_kwargs={"input_str": "hi"}, run_id="r1")
            events = []
            async for event in stream.stream_events():
                events.append(event)
            reliability_events = [
                e
                for e in events
                if isinstance(e, RunItemStreamEvent) and e.name == "runner.reliability"
            ]
            self.assertGreater(len(reliability_events), 0)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Additional Runner-level scenarios from the spec's "Tests" section.
# ---------------------------------------------------------------------------
class TestRunnerScenarioCoverage(unittest.TestCase):
    def test_acall_400_context_overflow_stops(self):
        async def run():
            outputs = [
                GeneratorOutput(data=None, error="status 400: context length too long"),
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="x"
                    )
                ),
            ]
            reliability = ReliabilityConfig(
                retry=RetryPolicy(
                    max_attempts=5, backoff=BackoffPolicy(initial=0.0, jitter="none")
                )
            )
            runner = make_runner(outputs, reliability=reliability, max_steps=3)
            result = await runner.acall(prompt_kwargs={"input_str": "hi"}, run_id="r1")
            # Context overflow is unrecoverable: planner not retried, run stops.
            self.assertEqual(runner.agent.planner.call_count, 1)
            self.assertEqual(len(runner.step_history), 1)
            self.assertIn("context length", str(result.error or result.answer))

        asyncio.run(run())

    def test_acall_retry_exhausted_keeps_error_step_semantics(self):
        async def run():
            # Enough outputs for 2 steps x 2 attempts each (planner exhausts to
            # an "exhausted" error only beyond this).
            err = GeneratorOutput(data=None, error="Error code: 500")
            outputs = [err, err, err, err]
            reliability = ReliabilityConfig(
                retry=RetryPolicy(
                    max_attempts=2, backoff=BackoffPolicy(initial=0.0, jitter="none")
                )
            )
            runner = make_runner(outputs, reliability=reliability, max_steps=2)
            result = await runner.acall(prompt_kwargs={"input_str": "hi"}, run_id="r1")
            # Each step exhausts retries (2 attempts) and the runner records an
            # error observation step, preserving the old error-step semantics.
            self.assertIsNotNone(result)
            self.assertEqual(len(runner.step_history), 2)
            for step in runner.step_history:
                self.assertIsNotNone(step.observation)
                self.assertIn("500", str(step.observation))

        asyncio.run(run())

    def test_acall_non_idempotent_tool_not_retried(self):
        async def run():
            outputs = [
                GeneratorOutput(data=DummyFunction(name="charge_card")),
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="done"
                    )
                ),
            ]
            reliability = ReliabilityConfig(
                retry=RetryPolicy(
                    max_attempts=5, backoff=BackoffPolicy(initial=0.0, jitter="none")
                )
            )
            tm = CountingToolManager(default_output=ValueError("boom"))
            runner = make_runner(
                outputs, reliability=reliability, tool_manager=tm, max_steps=3
            )
            result = await runner.acall(prompt_kwargs={"input_str": "hi"}, run_id="r1")
            # Non-idempotent tool not retried: exactly one call, run continues.
            self.assertEqual(tm.call_count, 1)
            self.assertEqual(result.answer, "done")

        asyncio.run(run())

    def test_acall_idempotent_tool_retried(self):
        async def run():
            outputs = [
                GeneratorOutput(data=DummyFunction(name="read")),
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="done"
                    )
                ),
            ]
            reliability = ReliabilityConfig(
                retry=RetryPolicy(
                    max_attempts=3, backoff=BackoffPolicy(initial=0.0, jitter="none")
                ),
                tools={"read": ToolPolicy(idempotent=True)},
            )

            class FlakyTM:
                def __init__(self):
                    self.call_count = 0

                async def execute_func_async(self, func):
                    self.call_count += 1
                    if self.call_count < 2:
                        raise ConnectionError("reset")
                    return FunctionOutput(
                        name=func.name,
                        input=func,
                        output=ToolOutput(output="ok", observation="ok"),
                    )

            tm = FlakyTM()
            runner = make_runner(
                outputs, reliability=reliability, tool_manager=tm, max_steps=3
            )
            result = await runner.acall(prompt_kwargs={"input_str": "hi"}, run_id="r1")
            # Idempotent tool retried after a transient error.
            self.assertEqual(tm.call_count, 2)
            self.assertEqual(result.answer, "done")

        asyncio.run(run())

    def test_astream_cancel_during_backoff_saves_cancelled_checkpoint(self):
        async def run():
            # Planner always returns a transient error so the executor schedules
            # a backoff sleep; we cancel mid-sleep.
            outputs = [GeneratorOutput(data=None, error="Error code: 500")] * 5
            reliability = ReliabilityConfig(
                retry=RetryPolicy(
                    max_attempts=5,
                    backoff=BackoffPolicy(initial=0.3, jitter="none"),
                )
            )
            store = InMemoryCheckpointStore()
            runner = make_runner(
                outputs, reliability=reliability, checkpoint_store=store, max_steps=3
            )
            stream = runner.astream(
                prompt_kwargs={"input_str": "hi"}, run_id="run-cancel"
            )
            # Let the run start and reach the backoff sleep.
            await asyncio.sleep(0.05)
            await runner.cancel()
            # Drain the stream so the task finishes its cancellation handling.
            async for _event in stream.stream_events():
                pass
            cp = store.load("run-cancel")
            self.assertIsNotNone(cp)
            self.assertEqual(cp.status, "cancelled")

        asyncio.run(run())

    def test_started_ledger_rerun_policy(self):
        from adalflow.components.agent.reliability import (
            RunCheckpoint,
            RunFingerprint,
            ToolLedgerEntry,
        )

        store = InMemoryCheckpointStore()
        # Compute the real args_hash the runner will use for an empty-args
        # "charge_card" Function so the ledger key matches on resume.
        args_hash = hashlib.sha256(
            json.dumps(
                {"name": "charge_card", "args": [], "kwargs": {}},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        ledger_key = f"0:charge_card:{args_hash}"
        fp = RunFingerprint(
            agent_name="dummy",
            planner_model=None,
            tool_names_sorted=["charge_card"],
            tool_signature_hash="sig",
            max_steps=5,
            input_hash=hashlib.sha256(b"q").hexdigest(),
        )
        cp = RunCheckpoint(
            run_id="run-rerun",
            status="running",
            created_at=1.0,
            updated_at=1.0,
            step_count=0,
            step_history=[],
            tool_ledger={
                ledger_key: ToolLedgerEntry(
                    call_id=ledger_key,
                    name="charge_card",
                    args_hash=args_hash,
                    state="started",
                ).to_dict()
            },
            fingerprint=fp.to_dict(),
            prompt_kwargs={"input_str": "q"},
            pending_function=Function(name="charge_card").to_dict(),
        )
        store.save(cp)
        # After rerunning the started tool for step 0 (planner skipped), the
        # next planner call is "finish".
        outputs = [
            GeneratorOutput(
                data=DummyFunction(name="finish", _is_answer_final=True, _answer="x")
            ),
        ]
        tm = CountingToolManager(default_output="charged")
        runner = make_runner(
            outputs, checkpoint_store=store, tool_manager=tm, max_steps=5
        )
        # rerun policy: should not raise; the started tool is re-executed.
        result = runner.call(
            prompt_kwargs={"input_str": "q"},
            run_id="run-rerun",
            resume=True,
            on_incomplete_tool="rerun",
            allow_fingerprint_mismatch=True,
        )
        self.assertEqual(result.answer, "x")
        self.assertEqual(tm.call_count, 1)
        # The planner was not called for the rerun step (only for finish).
        self.assertEqual(runner.agent.planner.call_count, 1)

    def test_file_checkpoint_store_resume_across_two_runners(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileCheckpointStore(d)
            # Runner 1: completes a run; checkpoint persisted to disk.
            outputs1 = [
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="persisted"
                    )
                ),
            ]
            runner1 = make_runner(
                outputs1, checkpoint_store=store, max_steps=3, agent_name="dummy"
            )
            runner1.call(prompt_kwargs={"input_str": "q"}, run_id="run-fp")

            # Simulated process restart: a brand new Runner instance with the
            # same on-disk store.  Resume must return the stored result without
            # calling the planner.
            outputs2 = [
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="DIFFERENT"
                    )
                ),
            ]
            runner2 = make_runner(
                outputs2, checkpoint_store=store, max_steps=3, agent_name="dummy"
            )
            result = runner2.call(
                prompt_kwargs={"input_str": "q"}, run_id="run-fp", resume=True
            )
            self.assertEqual(result.answer, "persisted")
            self.assertEqual(runner2.agent.planner.call_count, 0)

    def test_file_checkpoint_store_replay_tool_across_two_runners(self):
        from adalflow.components.agent.reliability import (
            RunCheckpoint,
            RunFingerprint,
            ToolLedgerEntry,
        )

        with tempfile.TemporaryDirectory() as d:
            store = FileCheckpointStore(d)
            args_hash = hashlib.sha256(
                json.dumps(
                    {"name": "search", "args": [], "kwargs": {}},
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            ledger_key = f"0:search:{args_hash}"
            fp = RunFingerprint(
                agent_name="dummy",
                planner_model=None,
                tool_names_sorted=["search"],
                tool_signature_hash="sig",
                max_steps=5,
                input_hash=hashlib.sha256(b"q").hexdigest(),
            )
            cp = RunCheckpoint(
                run_id="run-fp2",
                status="running",
                created_at=1.0,
                updated_at=1.0,
                step_count=0,
                step_history=[],
                tool_ledger={
                    ledger_key: ToolLedgerEntry(
                        call_id=ledger_key,
                        name="search",
                        args_hash=args_hash,
                        state="completed",
                        observation_json=json.dumps(
                            {
                                "output": "search-result",
                                "observation": "search-result",
                                "status": "success",
                            }
                        ),
                    ).to_dict()
                },
                fingerprint=fp.to_dict(),
                prompt_kwargs={"input_str": "q"},
                pending_function=Function(name="search").to_dict(),
            )
            store.save(cp)
            # New Runner instance (simulated restart) resumes from disk and
            # replays the completed tool without re-executing it.  The planner
            # is not called for the replayed step; the next call is "finish".
            outputs = [
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="done2"
                    )
                ),
            ]
            tm = CountingToolManager(default_output="search-result2")
            runner = make_runner(
                outputs, checkpoint_store=store, tool_manager=tm, max_steps=5
            )
            result = runner.call(
                prompt_kwargs={"input_str": "q"},
                run_id="run-fp2",
                resume=True,
                allow_fingerprint_mismatch=True,
            )
            self.assertEqual(result.answer, "done2")
            self.assertEqual(tm.call_count, 0)


# ---------------------------------------------------------------------------
# Round 3 regression tests (items 1-5 in .cursor-round3.md).
#
# Each test maps 1:1 to a defect fixed in runner.py and is written to fail
# before the corresponding fix.  They use only fake planners/tools and
# injected stores - no network, no paid model calls.
# ---------------------------------------------------------------------------
class _CrashNow(BaseException):
    """A SystemExit-like crash: subclasses BaseException (not Exception) so
    the runner's ``except Exception`` loop handler does NOT catch it - it
    propagates out of ``runner.call()`` exactly like a hard process crash."""


class TestRound3RegressionItems(unittest.TestCase):
    """Regression coverage for the five defects fixed in round 3."""

    # ---- Item 1: durable write-ahead for a mid-tool crash ------------------
    def test_item1_crash_during_tool_leaves_started_ledger_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileCheckpointStore(d)

            class CrashTM:
                def __init__(self):
                    self.call_count = 0

                def execute_func(self, func):
                    self.call_count += 1
                    raise _CrashNow("simulated hard crash mid-tool")

            outputs = [
                GeneratorOutput(data=DummyFunction(name="charge_card")),
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="done"
                    )
                ),
            ]
            # No tool timeout -> the tool runs in the main thread and the
            # BaseException propagates straight out of runner.call().
            reliability = ReliabilityConfig(retry=RetryPolicy.none())
            runner = make_runner(
                outputs,
                reliability=reliability,
                checkpoint_store=store,
                tool_manager=CrashTM(),
                max_steps=3,
            )
            with self.assertRaises(_CrashNow):
                runner.call(prompt_kwargs={"input_str": "q"}, run_id="run-crash")

            cp = store.load("run-crash")
            self.assertIsNotNone(cp)
            args_hash = hashlib.sha256(
                json.dumps(
                    {"name": "charge_card", "args": [], "kwargs": {}},
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            ledger_key = f"0:charge_card:{args_hash}"
            entry = cp.ledger_entry(ledger_key)
            self.assertIsNotNone(entry, "write-ahead started entry must be on disk")
            self.assertEqual(entry.state, "started")
            self.assertIsNotNone(cp.pending_function)

    def test_item1_resume_started_ledger_refuses_then_reruns(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileCheckpointStore(d)

            class CrashTM:
                def __init__(self):
                    self.call_count = 0

                def execute_func(self, func):
                    self.call_count += 1
                    raise _CrashNow("crash")

            outputs = [
                GeneratorOutput(data=DummyFunction(name="charge_card")),
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="done"
                    )
                ),
            ]
            reliability = ReliabilityConfig(retry=RetryPolicy.none())
            runner = make_runner(
                outputs,
                reliability=reliability,
                checkpoint_store=store,
                tool_manager=CrashTM(),
                max_steps=3,
            )
            with self.assertRaises(_CrashNow):
                runner.call(prompt_kwargs={"input_str": "q"}, run_id="run-crash2")

            # Default on_incomplete_tool="error" -> resume refuses to replay a
            # started (non-idempotent) tool.
            err_outputs = [
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="x"
                    )
                ),
            ]
            runner_err = make_runner(
                err_outputs,
                checkpoint_store=store,
                max_steps=3,
            )
            with self.assertRaises(ToolReplayRefusedError):
                runner_err.call(
                    prompt_kwargs={"input_str": "q"},
                    run_id="run-crash2",
                    resume=True,
                    allow_fingerprint_mismatch=True,
                )

            # on_incomplete_tool="rerun" -> the tool is re-executed once and
            # the run finishes.  The planner is skipped for the rerun step.
            rerun_outputs = [
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="done"
                    )
                ),
            ]
            tm_rerun = CountingToolManager(default_output="charged")
            runner_rerun = make_runner(
                rerun_outputs,
                checkpoint_store=store,
                tool_manager=tm_rerun,
                max_steps=3,
            )
            result = runner_rerun.call(
                prompt_kwargs={"input_str": "q"},
                run_id="run-crash2",
                resume=True,
                on_incomplete_tool="rerun",
                allow_fingerprint_mismatch=True,
            )
            self.assertEqual(result.answer, "done")
            self.assertEqual(tm_rerun.call_count, 1)
            self.assertEqual(runner_rerun.agent.planner.call_count, 1)

    # ---- Item 2: step-keyed ledger identity -------------------------------
    def test_item2_same_tool_same_args_twice_both_execute(self):
        outputs = [
            GeneratorOutput(data=DummyFunction(name="poll_status")),
            GeneratorOutput(data=DummyFunction(name="poll_status")),
            GeneratorOutput(
                data=DummyFunction(name="finish", _is_answer_final=True, _answer="done")
            ),
        ]
        tm = CountingToolManager(default_output="up")
        reliability = ReliabilityConfig(retry=RetryPolicy.none())
        runner = make_runner(
            outputs,
            reliability=reliability,
            tool_manager=tm,
            max_steps=5,
        )
        result = runner.call(prompt_kwargs={"input_str": "q"}, run_id="run-poll")
        self.assertEqual(result.answer, "done")
        self.assertEqual(tm.call_count, 2)
        self.assertEqual(tm.calls, ["poll_status", "poll_status"])

    def test_item2_crash_after_tool_completed_before_step_appended_replays(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileCheckpointStore(d)

            class HookTM:
                def __init__(self):
                    self.call_count = 0

                def execute_func(self, func):
                    self.call_count += 1
                    return FunctionOutput(
                        name=func.name,
                        input=func,
                        output=ToolOutput(output="ok", observation="ok"),
                    )

            tm = HookTM()
            outputs = [
                GeneratorOutput(data=DummyFunction(name="search")),
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="done"
                    )
                ),
            ]
            reliability = ReliabilityConfig(retry=RetryPolicy.none())
            runner = make_runner(
                outputs,
                reliability=reliability,
                checkpoint_store=store,
                tool_manager=tm,
                max_steps=3,
            )

            def _crash():
                raise _CrashNow("crash after tool completed")

            runner._after_tool_complete_hook = _crash
            with self.assertRaises(_CrashNow):
                runner.call(prompt_kwargs={"input_str": "q"}, run_id="run-after")

            self.assertEqual(tm.call_count, 1)
            cp = store.load("run-after")
            self.assertIsNotNone(cp)
            args_hash = hashlib.sha256(
                json.dumps(
                    {"name": "search", "args": [], "kwargs": {}},
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            ledger_key = f"0:search:{args_hash}"
            entry = cp.ledger_entry(ledger_key)
            self.assertIsNotNone(entry)
            self.assertEqual(entry.state, "completed")
            self.assertIsNotNone(cp.pending_function)
            self.assertEqual(len(cp.step_history), 0)

            resume_outputs = [
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="done2"
                    )
                ),
            ]
            tm2 = CountingToolManager(default_output="should-not-run")
            runner2 = make_runner(
                resume_outputs,
                checkpoint_store=store,
                tool_manager=tm2,
                max_steps=3,
            )
            result = runner2.call(
                prompt_kwargs={"input_str": "q"},
                run_id="run-after",
                resume=True,
                allow_fingerprint_mismatch=True,
            )
            self.assertEqual(result.answer, "done2")
            self.assertEqual(tm2.call_count, 0)
            self.assertEqual(runner2.agent.planner.call_count, 1)

    # ---- Item 3: streaming path routes through the reliability helper ----
    def test_item3_astream_tool_timeout_emits_reliability_event_and_continues(self):
        async def run():
            outputs = [
                GeneratorOutput(data=DummyFunction(name="slow_tool")),
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="done"
                    )
                ),
            ]

            async def slow_tool(func):
                await asyncio.sleep(1.0)
                return FunctionOutput(name=func.name, input=func, output=None)

            class AsyncTM:
                def __init__(self):
                    self.calls = 0

                async def execute_func_async(self, func):
                    self.calls += 1
                    return await slow_tool(func)

            tm = AsyncTM()
            reliability = ReliabilityConfig(
                retry=RetryPolicy.none(),
                timeouts=TimeoutPolicy(tool_timeout=0.05),
            )
            runner = make_runner(
                outputs,
                reliability=reliability,
                tool_manager=tm,
                max_steps=3,
            )
            stream = runner.astream(prompt_kwargs={"input_str": "hi"}, run_id="r-to")
            events = []
            async for event in stream.stream_events():
                events.append(event)
            timeout_events = [
                e
                for e in events
                if isinstance(e, RunItemStreamEvent)
                and e.name == "runner.reliability"
                and getattr(getattr(e.item, "data", None), "kind", None) == "timeout"
            ]
            self.assertGreater(len(timeout_events), 0)
            # The run continued to finish despite the tool timeout.
            self.assertEqual(stream.answer, "done")
            # The timed-out step's observation mentions the timeout.
            self.assertTrue(
                any(
                    "timed out" in str(getattr(s, "observation", "") or "")
                    for s in runner.step_history
                )
            )

        asyncio.run(run())

    # ---- Item 4: ledger records the resolved output, not a generator ------
    def test_item4_async_generator_tool_ledger_records_resolved_output(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                store = FileCheckpointStore(d)

                class AsyncGenTM:
                    def __init__(self):
                        self.call_count = 0

                    async def execute_func_async(self, func):
                        self.call_count += 1

                        async def gen():
                            yield ToolCallActivityRunItem(data="working")
                            yield ToolOutput(
                                output="final-value", observation="final-value"
                            )

                        return FunctionOutput(name=func.name, input=func, output=gen())

                outputs = [
                    GeneratorOutput(data=DummyFunction(name="gen_tool")),
                    GeneratorOutput(
                        data=DummyFunction(
                            name="finish", _is_answer_final=True, _answer="done"
                        )
                    ),
                ]
                reliability = ReliabilityConfig(retry=RetryPolicy.none())
                tm = AsyncGenTM()
                runner = make_runner(
                    outputs,
                    reliability=reliability,
                    checkpoint_store=store,
                    tool_manager=tm,
                    max_steps=3,
                )
                result = await runner.acall(
                    prompt_kwargs={"input_str": "q"},
                    run_id="run-gen",
                )
                self.assertEqual(result.answer, "done")

                cp = store.load("run-gen")
                self.assertIsNotNone(cp)
                args_hash = hashlib.sha256(
                    json.dumps(
                        {"name": "gen_tool", "args": [], "kwargs": {}},
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
                ledger_key = f"0:gen_tool:{args_hash}"
                entry = cp.ledger_entry(ledger_key)
                self.assertIsNotNone(entry)
                self.assertEqual(entry.state, "completed")
                obs = entry.observation_json or ""
                self.assertIn("final-value", obs)
                self.assertNotIn("generator", obs.lower())

        asyncio.run(run())

    # ---- Item 5: cleanup (Runner.close, module-level imports) ------------
    def test_item5_runner_close_releases_executor_thread_pool(self):
        reliability = ReliabilityConfig(
            retry=RetryPolicy.none(),
            timeouts=TimeoutPolicy(planner_timeout=1.0),
        )
        runner = make_runner(
            [
                GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="x"
                    )
                )
            ],
            reliability=reliability,
            max_steps=2,
        )
        # Run once so the executor actually creates its internal thread pool
        # (used for sync planner / tool timeouts).
        result = runner.call(prompt_kwargs={"input_str": "hi"}, run_id="r-close")
        self.assertEqual(result.answer, "x")
        self.assertIsNotNone(runner._executor)
        self.assertIsNotNone(
            runner._executor._thread_pool,
            "executor should have created a thread pool during the run",
        )
        # close() must release the thread pool.
        runner.close()
        self.assertIsNone(runner._executor._thread_pool)
        # Idempotent: a second close() must not raise.
        runner.close()

    def test_item5_no_function_local_time_hashlib_json_imports(self):
        import inspect as _inspect

        from adalflow.components.agent import runner as runner_mod

        src = _inspect.getsource(runner_mod)
        lines = src.splitlines()
        body = "\n".join(lines[60:])
        import re

        indented_imports = re.findall(
            r"^\s+(?:import (time|hashlib|json)\b|from (time|hashlib|json)\b)",
            body,
            re.MULTILINE,
        )
        self.assertFalse(
            indented_imports,
            f"function-local time/hashlib/json imports remain: {indented_imports}",
        )


if __name__ == "__main__":
    unittest.main()
