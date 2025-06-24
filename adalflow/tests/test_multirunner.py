import pytest
from unittest.mock import Mock, AsyncMock
from adalflow.core.multirunner import MultiRunner
from adalflow.core.runner import Runner
from adalflow.core.types import GeneratorOutput, StepOutput


class TestMultiRunner:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Create test data
        self.step_output = StepOutput(
            step=1, action="test", function=None, observation={"result": "test result"}
        )

        self.generator_output = GeneratorOutput(
            data=self.step_output, raw_response="raw response"
        )

        # Create mock runners
        self.runner1 = Mock(spec=Runner)
        self.runner1.call.return_value = ([self.generator_output], "result1")
        self.runner1.acall = AsyncMock(
            return_value=([self.generator_output], "async result1")
        )

        self.runner2 = Mock(spec=Runner)
        self.runner2.call.return_value = ([self.generator_output], "result2")
        self.runner2.acall = AsyncMock(
            return_value=([self.generator_output], "async result2")
        )

        # Create MultiRunner instance
        self.multi_runner = MultiRunner(
            runners={"runner1": self.runner1, "runner2": self.runner2}
        )

    # Test Initialization
    def test_init(self):
        """Test MultiRunner initialization"""
        assert "runner1" in self.multi_runner.runners
        assert "runner2" in self.multi_runner.runners
        assert self.multi_runner.get_runner("runner1") == self.runner1

    # Test Runner Management
    def test_add_runner(self):
        """Test adding a runner"""
        new_runner = Mock(spec=Runner)
        self.multi_runner.add_runner("new_runner", new_runner)
        assert "new_runner" in self.multi_runner.runners
        assert self.multi_runner.get_runner("new_runner") == new_runner

    def test_add_duplicate_runner(self):
        """Test adding a duplicate runner"""
        with pytest.raises(ValueError, match="Runner with name runner1 already exists"):
            self.multi_runner.add_runner("runner1", self.runner1)

    def test_remove_runner(self):
        """Test removing a runner"""
        self.multi_runner.remove_runner("runner1")
        assert "runner1" not in self.multi_runner.runners
        with pytest.raises(ValueError, match="Runner runner1 not found"):
            self.multi_runner.get_runner("runner1")

    # Test Call Methods
    def test_call(self):
        """Test calling a runner"""
        prompt_kwargs = {"input": "test"}

        # Call first runner
        step_history, result = self.multi_runner.call(
            "runner1",
            prompt_kwargs=prompt_kwargs,
            model_kwargs={"temperature": 0.7},
            use_cache=True,
            id="test1",
        )

        # Verify runner1 was called correctly
        self.runner1.call.assert_called_once_with(
            prompt_kwargs=prompt_kwargs,
            model_kwargs={"temperature": 0.7},
            use_cache=True,
            id="test1",
        )

        assert result == "result1"

        # Call second runner
        step_history, result = self.multi_runner.call("runner2", prompt_kwargs)
        assert result == "result2"

    @pytest.mark.asyncio
    async def test_acall(self):
        """Test async calling a runner"""
        prompt_kwargs = {"input": "async test"}

        # Call first runner
        step_history, result = await self.multi_runner.acall(
            "runner1",
            prompt_kwargs=prompt_kwargs,
            model_kwargs=None,
            use_cache=False,
            id="async_test",
        )

        # Verify runner1 was called correctly
        self.runner1.acall.assert_awaited_once_with(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=None,
            use_cache=False,
            id="async_test",
        )

        assert result == "async result1"

    # Test Error Cases
    def test_call_nonexistent_runner(self):
        """Test calling a non-existent runner"""
        with pytest.raises(ValueError, match="Runner nonexistent not found"):
            self.multi_runner.call("nonexistent", {"input": "test"})

    @pytest.mark.asyncio
    async def test_acall_runner_error(self):
        """Test error handling in async call"""
        self.runner1.acall.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            await self.multi_runner.acall("runner1", {"input": "test"})

    # Test Batch Operations
    def test_call_all(self):
        """Test calling all runners"""
        prompt_kwargs = {"input": "batch test"}

        results = self.multi_runner.call_all(
            prompt_kwargs=prompt_kwargs,
            model_kwargs={"batch": True},
            use_cache=True,
            id="batch_test",
        )

        assert len(results) == 2
        assert results["runner1"] == "result1"
        assert results["runner2"] == "result2"
