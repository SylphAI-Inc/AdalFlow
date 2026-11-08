"""Reliability layer unit tests: classifier, policy, checkpoint, store, executor.

No network, no paid model calls.  Uses injected clock/sleep where relevant.
"""

import asyncio
import json
import os
import tempfile
import unittest

from adalflow.components.agent.reliability import (
    BackoffPolicy,
    CheckpointCorruptedError,
    CheckpointVersionError,
    ErrorCategory,
    ErrorClassifier,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    ReliabilityConfig,
    ReliableExecutor,
    RetryPolicy,
    RunCheckpoint,
    RunFingerprint,
    TimeoutPolicy,
    ToolLedgerEntry,
    ToolPolicy,
    is_unrecoverable,
    register_migration,
)
from adalflow.components.agent.reliability.classification import ClassifiedError
from adalflow.core.types import (
    Function,
    FunctionOutput,
    GeneratorOutput,
    ToolOutput,
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class TestErrorClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = ErrorClassifier()

    def test_none(self):
        c = self.clf.classify(None)
        self.assertEqual(c.category, ErrorCategory.UNKNOWN)
        self.assertFalse(c.retryable)

    def test_1400_rows_false_positive(self):
        # "Processed 1400 rows" must NOT match status 400.
        c = self.clf.classify("Processed 1400 rows successfully")
        self.assertNotEqual(c.status_code, 400)
        # No HTTP-ish context keyword near the digits, so no status detected.
        self.assertIsNone(c.status_code)

    def test_status_code_in_text(self):
        c = self.clf.classify("Error code: 429")
        self.assertEqual(c.category, ErrorCategory.RATE_LIMITED)
        self.assertTrue(c.retryable)
        self.assertEqual(c.status_code, 429)

    def test_status_500_transient(self):
        c = self.clf.classify("HTTP status 500 internal server error")
        self.assertEqual(c.category, ErrorCategory.TRANSIENT)
        self.assertTrue(c.retryable)

    def test_context_overflow_phrase(self):
        c = self.clf.classify("This model's maximum context length is exceeded")
        self.assertEqual(c.category, ErrorCategory.CONTEXT_OVERFLOW)
        self.assertFalse(c.retryable)

    def test_400_with_context_phrase(self):
        c = self.clf.classify("status 400: context length too long")
        self.assertEqual(c.category, ErrorCategory.CONTEXT_OVERFLOW)
        self.assertFalse(c.retryable)

    def test_400_invalid_request(self):
        c = self.clf.classify("Error code: 400 bad request")
        self.assertEqual(c.category, ErrorCategory.INVALID_REQUEST)
        self.assertFalse(c.retryable)

    def test_404_not_found(self):
        c = self.clf.classify("Error code: 404 model not found")
        self.assertEqual(c.category, ErrorCategory.NOT_FOUND)
        self.assertFalse(c.retryable)

    def test_connection_error_phrase(self):
        c = self.clf.classify("Connection error: connection reset by peer")
        self.assertEqual(c.category, ErrorCategory.TRANSIENT)
        self.assertTrue(c.retryable)

    def test_rate_limit_phrase(self):
        c = self.clf.classify("Rate limit exceeded, please retry later")
        self.assertEqual(c.category, ErrorCategory.RATE_LIMITED)
        self.assertTrue(c.retryable)

    def test_retry_after_header(self):
        class FakeResp:
            status_code = 429
            headers = {"Retry-After": "5"}

        class FakeErr(Exception):
            response = FakeResp()

        c = self.clf.classify(FakeErr())
        self.assertEqual(c.category, ErrorCategory.RATE_LIMITED)
        self.assertEqual(c.retry_after, 5.0)

    def test_cause_chain(self):
        try:
            try:
                raise ConnectionError("reset")
            except ConnectionError as e:
                raise RuntimeError("wrapper") from e
        except RuntimeError as e:
            c = self.clf.classify(e)
            # Walks the cause chain and finds ConnectionError -> TRANSIENT.
            self.assertEqual(c.category, ErrorCategory.TRANSIENT)

    def test_cancelled_never_retryable(self):
        c = self.clf.classify(asyncio.CancelledError())
        self.assertEqual(c.category, ErrorCategory.CANCELLED)
        self.assertFalse(c.retryable)

    def test_user_rule_wins(self):
        self.clf.register(
            lambda e: isinstance(e, str) and "boom" in e,
            ErrorCategory.INVALID_REQUEST,
            False,
        )
        c = self.clf.classify("boom rate limit")
        self.assertEqual(c.category, ErrorCategory.INVALID_REQUEST)

    def test_timeout_phrase_is_timeout_not_transient(self):
        # "timeout"/"timed out" phrases must map to TIMEOUT, not TRANSIENT,
        # so they do not shadow the dedicated timeout category.
        c = self.clf.classify("the request timed out waiting for a response")
        self.assertEqual(c.category, ErrorCategory.TIMEOUT)
        self.assertTrue(c.retryable)
        c2 = self.clf.classify("timeout while calling the model")
        self.assertEqual(c2.category, ErrorCategory.TIMEOUT)

    def test_bare_429_substring_not_matched(self):
        # A bare "429" appearing in arbitrary text must NOT be treated as a
        # rate-limit status code (substring digit matching is the bug).
        c = self.clf.classify("Processed 429 rows in batch 429")
        self.assertIsNone(c.status_code)
        self.assertEqual(c.category, ErrorCategory.UNKNOWN)

    def test_file_not_found_is_not_transient(self):
        # FileNotFoundError is an OSError but not a network error: it must
        # not be auto-classified as TRANSIENT.
        c = self.clf.classify(FileNotFoundError("missing.txt"))
        self.assertNotEqual(c.category, ErrorCategory.TRANSIENT)

    def test_connection_reset_errno_is_transient(self):
        import errno as _errno

        c = self.clf.classify(OSError(_errno.ECONNRESET, "Connection reset"))
        self.assertEqual(c.category, ErrorCategory.TRANSIENT)
        self.assertTrue(c.retryable)

    def test_generic_outer_exception_walks_chain_first(self):
        # An outer generic exception must not be classified via its own
        # message as UNKNOWN before the chain is walked.
        try:
            try:
                raise ConnectionError("reset")
            except ConnectionError as e:
                raise ValueError("tool failed") from e
        except ValueError as e:
            c = self.clf.classify(e)
            self.assertEqual(c.category, ErrorCategory.TRANSIENT)

    def test_is_unrecoverable(self):
        self.assertTrue(
            is_unrecoverable(
                ClassifiedError(
                    category=ErrorCategory.CONTEXT_OVERFLOW, retryable=False
                )
            )
        )
        self.assertFalse(
            is_unrecoverable(
                ClassifiedError(category=ErrorCategory.TRANSIENT, retryable=True)
            )
        )
        self.assertTrue(
            is_unrecoverable(
                ClassifiedError(category=ErrorCategory.CANCELLED, retryable=False)
            )
        )


# ---------------------------------------------------------------------------
# Backoff / RetryPolicy
# ---------------------------------------------------------------------------
class TestBackoffPolicy(unittest.TestCase):
    def test_no_jitter(self):
        b = BackoffPolicy(initial=1.0, multiplier=2.0, max_delay=10.0, jitter="none")
        self.assertAlmostEqual(b.delay(0), 1.0)
        self.assertAlmostEqual(b.delay(1), 2.0)
        self.assertAlmostEqual(b.delay(2), 4.0)
        self.assertAlmostEqual(b.delay(10), 10.0)  # capped

    def test_retry_after_overrides(self):
        b = BackoffPolicy(initial=1.0, multiplier=2.0, max_delay=10.0)
        self.assertAlmostEqual(b.delay(0, retry_after=3.0), 3.0)
        self.assertAlmostEqual(b.delay(0, retry_after=100.0), 10.0)

    def test_full_jitter_bounds(self):
        import random

        rng = random.Random(42)
        b = BackoffPolicy(initial=2.0, multiplier=1.0, max_delay=10.0, jitter="full")
        for _ in range(50):
            d = b.delay(0, rng=rng)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 2.0)

    def test_equal_jitter_bounds(self):
        import random

        rng = random.Random(42)
        b = BackoffPolicy(initial=4.0, multiplier=1.0, max_delay=10.0, jitter="equal")
        for _ in range(50):
            d = b.delay(0, rng=rng)
            self.assertGreaterEqual(d, 2.0)
            self.assertLessEqual(d, 4.0)


class TestRetryPolicy(unittest.TestCase):
    def test_none_policy(self):
        p = RetryPolicy.none()
        self.assertEqual(p.max_attempts, 1)
        c = ClassifiedError(category=ErrorCategory.TRANSIENT, retryable=True)
        self.assertFalse(p.should_retry(c, 0))

    def test_validation(self):
        with self.assertRaises(ValueError):
            RetryPolicy(max_attempts=0)

    def test_should_retry(self):
        p = RetryPolicy(max_attempts=3)
        c = ClassifiedError(category=ErrorCategory.TRANSIENT, retryable=True)
        self.assertTrue(p.should_retry(c, 0))
        self.assertTrue(p.should_retry(c, 1))
        self.assertFalse(p.should_retry(c, 2))

    def test_max_total_delay(self):
        p = RetryPolicy(max_attempts=10, max_total_delay=1.0)
        c = ClassifiedError(category=ErrorCategory.TRANSIENT, retryable=True)
        self.assertFalse(p.should_retry(c, 0, elapsed_delay=2.0))


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class FakeClock:
    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeAsyncSleep:
    def __init__(self, clock):
        self.clock = clock
        self.sleeps = []

    async def __call__(self, delay):
        self.sleeps.append(delay)
        self.clock.advance(delay)


def make_executor(
    retry=None, timeouts=None, clock=None, sleep=None, on_event=None, seed=42
):
    config = ReliabilityConfig(
        retry=retry or RetryPolicy.none(),
        timeouts=timeouts or TimeoutPolicy(),
        seed=seed,
    )
    clk = clock or FakeClock()
    slp = sleep or FakeAsyncSleep(clk)
    return ReliableExecutor(config=config, clock=clk, sleep=slp, on_event=on_event)


class TestReliableExecutor(unittest.TestCase):
    def test_planner_retry_success_after_n(self):
        async def run():
            clk = FakeClock()
            slp = FakeAsyncSleep(clk)
            retry = RetryPolicy(
                max_attempts=3, backoff=BackoffPolicy(initial=0.0, jitter="none")
            )
            ex = make_executor(retry=retry, clock=clk, sleep=slp)
            ex.start_run()
            calls = {"n": 0}

            async def fn():
                calls["n"] += 1
                if calls["n"] < 3:
                    return GeneratorOutput(data=None, error="Error code: 500")
                return GeneratorOutput(data=Function(name="finish"))

            out = await ex.run_planner(fn, step=0)
            self.assertIsNotNone(out.data)
            self.assertEqual(calls["n"], 3)

        asyncio.run(run())

    def test_429_retried_not_terminated(self):
        async def run():
            retry = RetryPolicy(
                max_attempts=4, backoff=BackoffPolicy(initial=0.0, jitter="none")
            )
            ex = make_executor(retry=retry)
            ex.start_run()
            calls = {"n": 0}

            async def fn():
                calls["n"] += 1
                if calls["n"] < 3:
                    return GeneratorOutput(data=None, error="Error code: 429")
                return GeneratorOutput(data=Function(name="finish"))

            out = await ex.run_planner(fn, step=0)
            self.assertIsNotNone(out.data)
            self.assertEqual(calls["n"], 3)

        asyncio.run(run())

    def test_non_retryable_400_context_overflow_stops(self):
        async def run():
            retry = RetryPolicy(
                max_attempts=5, backoff=BackoffPolicy(initial=0.0, jitter="none")
            )
            ex = make_executor(retry=retry)
            ex.start_run()
            calls = {"n": 0}

            async def fn():
                calls["n"] += 1
                return GeneratorOutput(
                    data=None, error="status 400: context length too long"
                )

            out = await ex.run_planner(fn, step=0)
            self.assertIsNone(out.data)
            self.assertEqual(calls["n"], 1)

        asyncio.run(run())

    def test_retry_exhausted_returns_last_output(self):
        async def run():
            retry = RetryPolicy(
                max_attempts=2, backoff=BackoffPolicy(initial=0.0, jitter="none")
            )
            ex = make_executor(retry=retry)
            ex.start_run()

            async def fn():
                return GeneratorOutput(data=None, error="Error code: 500")

            out = await ex.run_planner(fn, step=0)
            self.assertIsNone(out.data)
            self.assertIn("500", out.error)

        asyncio.run(run())

    def test_planner_timeout(self):
        async def run():
            clk = FakeClock()
            slp = FakeAsyncSleep(clk)
            retry = RetryPolicy(
                max_attempts=1, backoff=BackoffPolicy(initial=0.0, jitter="none")
            )
            timeouts = TimeoutPolicy(planner_timeout=0.1)
            ex = make_executor(retry=retry, timeouts=timeouts, clock=clk, sleep=slp)
            ex.start_run()

            async def fn():
                await asyncio.sleep(1.0)
                return GeneratorOutput(data=Function(name="finish"))

            out = await ex.run_planner(fn, step=0)
            self.assertIsNone(out.data)
            self.assertIn("timed out", out.error)

        asyncio.run(run())

    def test_tool_timeout_returns_timeout_observation(self):
        async def run():
            clk = FakeClock()
            slp = FakeAsyncSleep(clk)
            retry = RetryPolicy.none()
            timeouts = TimeoutPolicy(tool_timeout=0.1)
            ex = make_executor(retry=retry, timeouts=timeouts, clock=clk, sleep=slp)
            ex.start_run()

            async def fn():
                await asyncio.sleep(1.0)
                return FunctionOutput(name="slow", input=None, output=None)

            func = Function(name="slow", id="c1")
            out = await ex.run_tool(fn, func, step=0)
            self.assertIsInstance(out.output, ToolOutput)
            self.assertEqual(out.output.status, "timeout")
            self.assertIn("timed out", out.output.observation)

        asyncio.run(run())

    def test_non_idempotent_tool_not_retried(self):
        async def run():
            retry = RetryPolicy(
                max_attempts=5, backoff=BackoffPolicy(initial=0.0, jitter="none")
            )
            ex = make_executor(retry=retry)
            ex.start_run()
            calls = {"n": 0}

            async def fn():
                calls["n"] += 1
                raise ValueError("boom")

            func = Function(name="charge_card", id="c1")
            out = await ex.run_tool(fn, func, step=0)
            self.assertEqual(calls["n"], 1)
            self.assertEqual(out.output.status, "error")

        asyncio.run(run())

    def test_idempotent_tool_retried(self):
        async def run():
            retry = RetryPolicy(
                max_attempts=3, backoff=BackoffPolicy(initial=0.0, jitter="none")
            )
            config = ReliabilityConfig(
                retry=retry, tools={"read": ToolPolicy(idempotent=True)}, seed=42
            )
            clk = FakeClock()
            slp = FakeAsyncSleep(clk)
            ex = ReliableExecutor(config=config, clock=clk, sleep=slp)
            ex.start_run()
            calls = {"n": 0}

            async def fn():
                calls["n"] += 1
                if calls["n"] < 2:
                    raise ConnectionError("reset")
                return FunctionOutput(
                    name="read",
                    input=None,
                    output=ToolOutput(output="ok", observation="ok"),
                )

            func = Function(name="read", id="c1")
            out = await ex.run_tool(fn, func, step=0)
            self.assertEqual(calls["n"], 2)
            self.assertEqual(out.output.observation, "ok")

        asyncio.run(run())

    def test_run_deadline_exceeded(self):
        async def run():
            clk = FakeClock()
            slp = FakeAsyncSleep(clk)
            retry = RetryPolicy(
                max_attempts=3, backoff=BackoffPolicy(initial=0.0, jitter="none")
            )
            timeouts = TimeoutPolicy(run_timeout=0.0)
            ex = make_executor(retry=retry, timeouts=timeouts, clock=clk, sleep=slp)
            ex.start_run()

            async def fn():
                return GeneratorOutput(data=None, error="Error code: 500")

            from adalflow.components.agent.reliability.errors import (
                RunDeadlineExceededError,
            )

            with self.assertRaises(RunDeadlineExceededError):
                await ex.run_planner(fn, step=0)

        asyncio.run(run())

    def test_cancellation_during_backoff_propagates(self):
        async def run():
            clk = FakeClock()
            slp = FakeAsyncSleep(clk)
            retry = RetryPolicy(
                max_attempts=5, backoff=BackoffPolicy(initial=1.0, jitter="none")
            )
            ex = make_executor(retry=retry, clock=clk, sleep=slp)
            ex.start_run()

            async def real_sleep(delay):
                raise asyncio.CancelledError()

            ex._sleep = real_sleep

            async def fn():
                return GeneratorOutput(data=None, error="Error code: 500")

            with self.assertRaises(asyncio.CancelledError):
                await ex.run_planner(fn, step=0)

        asyncio.run(run())

    def test_stats(self):
        async def run():
            retry = RetryPolicy(
                max_attempts=3, backoff=BackoffPolicy(initial=0.0, jitter="none")
            )
            ex = make_executor(retry=retry)
            ex.start_run()
            calls = {"n": 0}

            async def fn():
                calls["n"] += 1
                if calls["n"] < 2:
                    return GeneratorOutput(data=None, error="Error code: 500")
                return GeneratorOutput(data=Function(name="finish"))

            await ex.run_planner(fn, step=0)
            stats = ex.stats()
            self.assertGreaterEqual(stats["planner:step0"]["attempts"], 2)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
class TestRunCheckpoint(unittest.TestCase):
    def test_json_round_trip(self):
        cp = RunCheckpoint(
            run_id="run-1",
            status="running",
            created_at=1.0,
            updated_at=2.0,
            step_count=3,
            step_history=[{"step": 0, "observation": "x"}],
            tool_ledger={
                "c1": ToolLedgerEntry(
                    call_id="c1",
                    name="search",
                    args_hash="h",
                    state="completed",
                    observation_json='{"observation": "ok"}',
                ).to_dict()
            },
            fingerprint=RunFingerprint(
                agent_name="A",
                planner_model="gpt",
                tool_names_sorted=["search"],
                tool_signature_hash="sig",
                max_steps=10,
                input_hash="ih",
            ).to_dict(),
            token_consumption={"total_prompt_tokens": 5},
            answer=None,
            error=None,
            prompt_kwargs={"input_str": "hi"},
            metadata={},
        )
        text = cp.to_json()
        cp2 = RunCheckpoint.from_json(text)
        self.assertEqual(cp2.run_id, "run-1")
        self.assertEqual(cp2.step_count, 3)
        self.assertEqual(cp2.tool_ledger["c1"]["state"], "completed")
        self.assertEqual(cp2.fingerprint_obj().agent_name, "A")

    def test_sha_mismatch_corrupted(self):
        cp = RunCheckpoint(
            run_id="run-2", status="running", created_at=1.0, updated_at=1.0
        )
        text = cp.to_json()
        envelope = json.loads(text)
        envelope["payload"]["step_count"] = 99
        tampered = json.dumps(envelope)
        with self.assertRaises(CheckpointCorruptedError):
            RunCheckpoint.from_json(tampered)

    def test_version_error_and_migration(self):
        cp = RunCheckpoint(
            run_id="run-3", status="running", created_at=1.0, updated_at=1.0
        )
        text = cp.to_json()
        envelope = json.loads(text)
        envelope["payload"]["schema_version"] = 0
        envelope["sha256"] = None  # skip integrity check for this test
        text_v0 = json.dumps(envelope)
        with self.assertRaises(CheckpointVersionError):
            RunCheckpoint.from_json(text_v0)

        register_migration(0, lambda data: {**data, "schema_version": 1})
        cp2 = RunCheckpoint.from_json(text_v0)
        self.assertEqual(cp2.schema_version, 1)


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------
class TestInMemoryCheckpointStore(unittest.TestCase):
    def test_save_load_delete(self):
        store = InMemoryCheckpointStore()
        cp = RunCheckpoint(
            run_id="r1", status="running", created_at=1.0, updated_at=1.0
        )
        store.save(cp)
        loaded = store.load("r1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "r1")
        loaded.status = "completed"
        again = store.load("r1")
        self.assertEqual(again.status, "running")
        store.delete("r1")
        self.assertIsNone(store.load("r1"))

    def test_list_runs(self):
        store = InMemoryCheckpointStore()
        store.save(
            RunCheckpoint(run_id="a", status="running", created_at=1.0, updated_at=1.0)
        )
        store.save(
            RunCheckpoint(
                run_id="b", status="completed", created_at=2.0, updated_at=2.0
            )
        )
        self.assertEqual(len(store.list_runs()), 2)
        self.assertEqual(len(store.list_runs(status="completed")), 1)


class TestFileCheckpointStore(unittest.TestCase):
    def test_atomic_write_previous_intact_on_crash(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileCheckpointStore(d)
            cp1 = RunCheckpoint(
                run_id="r1",
                status="running",
                created_at=1.0,
                updated_at=1.0,
                step_count=1,
            )
            store.save(cp1)
            orig = store._atomic_write

            def boom(path, data):
                raise RuntimeError("crash")

            store._atomic_write = boom
            with self.assertRaises(RuntimeError):
                store.save(
                    RunCheckpoint(
                        run_id="r1",
                        status="running",
                        created_at=2.0,
                        updated_at=2.0,
                        step_count=2,
                    )
                )
            store._atomic_write = orig
            loaded = store.load("r1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.step_count, 1)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileCheckpointStore(d)
            store.save(
                RunCheckpoint(
                    run_id="../escape", status="running", created_at=1.0, updated_at=1.0
                )
            )
            files = [f for f in os.listdir(d) if f.endswith(".json")]
            self.assertEqual(len(files), 1)
            self.assertNotIn("..", files[0])

    def test_history_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileCheckpointStore(d, keep_history=2)
            store.save(
                RunCheckpoint(
                    run_id="r1",
                    status="running",
                    created_at=1.0,
                    updated_at=1.0,
                    step_count=1,
                )
            )
            store.save(
                RunCheckpoint(
                    run_id="r1",
                    status="running",
                    created_at=2.0,
                    updated_at=2.0,
                    step_count=2,
                )
            )
            primary = os.path.join(d, "r1.json")
            with open(primary, "w") as f:
                f.write("not json")
            loaded = store.load("r1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.step_count, 1)
