"""
title: Compute Pipeline with Bash Tool
author: Assistant
date: 2024-11-20
version: 1.0
license: MIT
description: A pipeline that enables execution of bash commands through an API endpoint
environment_variables: ANTHROPIC_API_KEY
"""

import os
import logging
import asyncio
from typing import List, Dict, Union, Generator, Iterator
from pydantic import BaseModel
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
stream_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(stream_handler)


def detect_code_fence(code_snippet: str) -> str:
    """
    Detect the language of a code snippet and wrap it in appropriate code fences.

    Args:
        code_snippet: The code or text to analyze

    Returns:
        The input wrapped in language-specific code fences
    """
    if not code_snippet.strip():
        return "\n```\n\n```\n"

    try:
        lexer = guess_lexer(code_snippet.strip())
        return f"\n```{lexer.name.lower()}\n{code_snippet}\n```\n"
    except ClassNotFound:
        return f"\n```plain\n{code_snippet}\n```\n"


class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""

    def __init__(self):
        self.name = "Compute Pipeline"
        self.type = "manifold"
        self.id = "compute"
        logger.debug(f"#### Initializing {self.name} pipeline")

        self.valves = self.Valves(
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
        )

    async def on_startup(self):
        logger.info(f"on_startup:{__name__}")

    async def on_shutdown(self):
        logger.info(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        logger.info(f"on_valves_updated:{__name__}")

    def pipelines(self) -> List[dict]:
        """Return list of supported models/pipelines"""
        return [
            {
                "id": "compute-bash",
                "name": "Compute Pipeline (Bash)",
                "description": "Execute bash commands via pipeline",
            }
        ]

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Execute bash commands through Claude with tool integration."""
        logger.info(f"pipe called with user_message: {user_message}")

        from dev.compute.loop import sampling_loop, APIProvider

        # Handle title request
        if body.get("title", False):
            return "Compute Pipeline"

        # Verify the model is supported
        if model_id != "compute-bash":
            error_msg = f"Unsupported model ID: {model_id}. Use 'compute-bash'."
            logger.error(error_msg)
            return error_msg

        try:
            # Create a new event loop for this request
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def process_message():
                # Extract system message and format messages
                formatted_messages = []
                system_prompt_suffix = ""

                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]

                    # Extract system message
                    if role == "system" and isinstance(content, str):
                        system_prompt_suffix = content
                        continue

                    if isinstance(content, str):
                        formatted_messages.append(
                            {
                                "role": role,
                                "content": [{"type": "text", "text": content}],
                            }
                        )
                    else:
                        formatted_messages.append({"role": role, "content": content})

                # Add current user message
                formatted_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_message}],
                    }
                )

                class StreamingContext:
                    def __init__(self):
                        self._queue = asyncio.Queue()
                        logger.info("StreamingContext initialized")

                    def output_callback(self, content: Dict):
                        logger.info(f"Output callback received: {content}")
                        if content["type"] == "text":
                            logger.info(f"Queueing text: {content['text']}")
                            self._queue.put_nowait(content["text"])

                    def tool_callback(self, result, tool_id):
                        logger.info(f"Tool callback received: {result}, {tool_id}")

                        outputs = []
                        if hasattr(result, "error") and result.error:
                            if hasattr(result, "exit_code") and result.exit_code:
                                outputs.append(f"Exit code: {result.exit_code}")
                            outputs.append(detect_code_fence(result.error))
                        if hasattr(result, "output") and result.output:
                            outputs.append(f"\n{detect_code_fence(result.output)}")

                        logger.info(f"Tool callback queueing outputs: {outputs}")
                        for output in outputs:
                            self._queue.put_nowait(output)

                    async def get(self):
                        try:
                            return await self._queue.get()
                        except asyncio.CancelledError:
                            return None

                ctx = StreamingContext()

                # Start the sampling loop in the background
                task = asyncio.create_task(
                    sampling_loop(
                        model="claude-3-5-sonnet-20241022",
                        provider=APIProvider.ANTHROPIC,
                        system_prompt_suffix=system_prompt_suffix,
                        messages=formatted_messages,
                        output_callback=ctx.output_callback,
                        tool_output_callback=ctx.tool_callback,
                        api_response_callback=lambda req, res, exc: None,
                        api_key=self.valves.ANTHROPIC_API_KEY,
                    )
                )

                try:
                    while not task.done() or not ctx._queue.empty():
                        try:
                            output = await asyncio.wait_for(ctx.get(), timeout=0.1)
                            if output:
                                yield output
                        except asyncio.TimeoutError:
                            # No output available, check if task is done
                            continue
                        except Exception as e:
                            logger.error(f"Error getting output: {e}")
                            break
                finally:
                    # Make sure we complete the task
                    if not task.done():
                        task.cancel()
                    try:
                        await task
                    except Exception as e:
                        logger.error(f"Error completing task: {e}")

            async def run_generator():
                logger.info("Starting generator")
                async for msg in process_message():
                    logger.info(f"Generated message: {msg}")
                    yield msg
                logger.info("Generator completed")

            if body.get("stream", False):
                logger.info("Streaming mode enabled")

                def sync_generator():
                    async_gen = run_generator()
                    while True:
                        try:
                            item = loop.run_until_complete(async_gen.__anext__())
                            logger.info(f"Streaming item: {item}")
                            yield item
                        except StopAsyncIteration:
                            logger.info("Streaming complete")
                            break
                    loop.close()

                return sync_generator()
            else:
                # For non-streaming, collect all output and return as string
                logger.info("Non-streaming mode")
                output_parts = []

                async def collect_output():
                    async for msg in run_generator():
                        logger.info(f"Collecting message: {msg}")
                        output_parts.append(msg)

                loop.run_until_complete(collect_output())
                loop.close()
                result = (
                    "\n".join(output_parts)
                    if output_parts
                    else "Command executed successfully"
                )
                logger.info(f"Non-streaming result: {result}")
                return result

        except Exception as e:
            error_msg = f"Error in pipe: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
