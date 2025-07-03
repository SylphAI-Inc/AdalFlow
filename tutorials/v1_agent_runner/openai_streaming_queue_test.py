import asyncio
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    List,
    Optional,
    AsyncIterable,
)

from adalflow.components.model_client.openai_client import OpenAIClient
from pydantic import BaseModel
from dotenv import load_dotenv
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseTextDeltaEvent,
)
import aioitertools
from openai import AsyncOpenAI
from adalflow.core.types import ModelType

import logging

log = logging.getLogger(__name__)

load_dotenv()


def parse_stream_response(event) -> str:
    """
    Extract the text fragment from a single SSE event of the Responses API.
    Returns the chunk if it's a delta or a done event, else an empty string.
    """
    # incremental text tokens
    if isinstance(event, ResponseTextDeltaEvent):
        return event.delta

    return ""


async def handle_streaming_response(
    stream: AsyncIterable,
) -> AsyncIterator[str]:
    """
    Iterate over an SSE stream from client.responses.create(..., stream=True),
    logging each raw event and yielding non-empty text fragments.
    """
    async for event in stream:
        log.debug(f"Raw event: {event!r}")
        text = parse_stream_response(event)
        if text:
            yield text


async def collect_final_response_from_stream(stream: AsyncIterable) -> str:
    """
    Collect the final complete response text from a streaming Response API.
    Consumes the entire stream and returns the concatenated result.
    """
    final_text = ""
    async for event in stream:
        log.debug(f"Raw event: {event!r}")

        # --- final completion event? ---
        if isinstance(event, ResponseCompletedEvent):
            resp = event.response
            log.debug(f"Response completed: {event.response.output_text}")
            # 1) old convenience property
            if getattr(resp, "output_text", None):
                return resp.output_text

        # --- intermediate delta event: accumulate via your parser ---
        text = parse_stream_response(event)
        if text:
            final_text += text

    # if we ran out of events without a ResponseCompletedEvent
    return final_text


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
                # self._event_queue.put_nowait(StreamEvent(type="chunk", data=chunk))
                self._event_queue.put_nowait(chunk)
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
    # stream = await client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "Extract entities from the input text"},
    #         # another text but similar prompt to test the queue
    #         {
    #             "role": "user",
    #             "content": "The quick brown fox jumps over the lazy dog with piercing blue eyes",
    #         },
    #         # {"role": "user", "content": "A swift blue jay soars over the peaceful meadow at dawn's first light"},
    #         # avoid prompt caching
    #     ],
    #     stream=True,
    # )
    # Test 1: handle_streaming_response
    print("Testing handle_streaming_response...")
    # use
    stream1 = await client.responses.create(
        model="gpt-4",
        # input=[
        #     {"role": "system", "content": "Extract entities from the input text"},
        #     {
        #         "role": "user",
        #         "content": "The quick brown fox jumps over the lazy dog with piercing blue eyes",
        #     },
        # ],
        input="Help me analyze a customer satisfaction survey. What steps should I take?",
        stream=True,
    )

    stream_1, stream_2 = aioitertools.tee(stream1, 2)

    print("Chunks from handle_streaming_response:")
    async for event in handle_streaming_response(stream_1):
        print(f"Chunk: {event}")

    print("Final response from collect_final_response_from_stream:")
    final_response = await collect_final_response_from_stream(stream_2)
    print(f"Final response: {final_response}")

    # Start processing the stream in the background
    # result._run_task = asyncio.create_task(result._process_stream(stream))

    print(f"Queue-based streaming completed in {time.time() - start_time:.4f} seconds")
    result.cancel()


async def adalflow_streaming():
    """Stream responses using AdalFlow OpenAI client."""
    client = OpenAIClient()
    # start_time = time.time()

    print("\n--- AdalFlow Streaming ---")

    # Use the specified input format
    stream_1 = await client.acall(
        api_kwargs={
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well—thanks!"},
            ],
            "stream": True,
        },
        model_type=ModelType.LLM,
    )

    async for event in stream_1:
        if type(event) == ResponseCompletedEvent:
            print(event.response.output_text)

    print("stream")

    async for event in stream_1:
        print(f"Chunk: {event}")

    # stream_1, stream_2 = aioitertools.tee(stream1, 2)

    # print("Chunks from handle_streaming_response:")
    # async for event in handle_streaming_response(stream_1):
    #     print(f"Chunk: {event}")

    # print("Final response from collect_final_response_from_stream:")
    # final_response = await collect_final_response_from_stream(stream_2)
    # print(f"Final response: {final_response}")

    # print(f"AdalFlow streaming completed in {time.time() - start_time:.4f} seconds")


async def main():
    # Run direct streaming
    # await direct_streaming()

    # Run queue-based streaming
    # await queue_based_streaming()

    # Run AdalFlow streaming
    await adalflow_streaming()

    # await direct_streaming()


if __name__ == "__main__":
    asyncio.run(main())
