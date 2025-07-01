import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class EntitiesModel(BaseModel):
    attributes: List[str]
    colors: List[str]
    animals: List[str]


@dataclass
class StreamEvent:
    type: str
    data: Any


@dataclass
class QueueCompleteSentinel:
    pass


class OpenAIStreamingResult:
    def __init__(self):
        self._event_queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel] = (
            asyncio.Queue()
        )
        self._run_task: Optional[asyncio.Task] = None
        self._exception: Optional[Exception] = None

    async def _process_stream(self, stream):
        try:
            async for chunk in stream:
                self._event_queue.put_nowait(StreamEvent(type="chunk", data=chunk))
            self._event_queue.put_nowait(QueueCompleteSentinel())
        except Exception as e:
            self._exception = e
            self._event_queue.put_nowait(QueueCompleteSentinel())

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        while True:
            if self._exception:
                raise self._exception

            try:
                item = await self._event_queue.get()
                if isinstance(item, QueueCompleteSentinel):
                    self._event_queue.task_done()
                    break

                yield item
                self._event_queue.task_done()
            except asyncio.CancelledError:
                break

    def cancel(self):
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()


async def direct_streaming():
    client = AsyncOpenAI()
    start_time = time.time()

    print("\n--- Direct Streaming ---")
    # Create a streaming chat completion
    stream = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Extract entities from the input text"},
            {
                "role": "user",
                "content": "A swift blue jay soars over the peaceful meadow at dawn's first light",
            },
        ],
        stream=True,
    )

    # Async iterate over the stream
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(f"[Direct] content: {chunk.choices[0].delta.content}")

    print(f"Direct streaming completed in {time.time() - start_time:.4f} seconds")


async def queue_based_streaming():
    """Stream responses from OpenAI using a queue-based approach."""
    client = AsyncOpenAI()
    start_time = time.time()

    print("\n--- Queue-based Streaming ---")

    # Create streaming result
    result = OpenAIStreamingResult()

    # Start the stream in the background
    stream = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Extract entities from the input text"},
            # another text but similar prompt to test the queue
            {
                "role": "user",
                "content": "The quick brown fox jumps over the lazy dog with piercing blue eyes",
            },
            # {"role": "user", "content": "A swift blue jay soars over the peaceful meadow at dawn's first light"},
            # avoid prompt caching
        ],
        stream=True,
    )

    # Start processing the stream in the background
    result._run_task = asyncio.create_task(result._process_stream(stream))

    # Process events from the queue
    async for event in result.stream_events():
        content = (
            event.data.choices[0].delta.content
            if event.data.choices and event.data.choices[0].delta.content
            else ""
        )
        print(f"[Queue] chunk: {content}")

    # Cleanup
    print(f"Queue-based streaming completed in {time.time() - start_time:.4f} seconds")
    result.cancel()


async def main():
    # Run direct streaming
    # await direct_streaming()

    # Run queue-based streaming
    await queue_based_streaming()

    await direct_streaming()


if __name__ == "__main__":
    asyncio.run(main())
