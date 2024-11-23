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
from typing import List, Dict
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dev.compute.loop import execute_command


class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""

    def __init__(self):
        self.name = "Compute Pipeline"
        self.type = "manifold"
        self.id = "compute"
        logger.debug(f"#### Initializing {self.name} pipeline")

        # Initialize valves
        self.valves = self.Valves(ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""))

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        print(f"on_valves_updated:{__name__}")

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
    ) -> str:
        """Execute bash commands through Claude with tool integration."""
        logger.info(f"pipe called with user_message: {user_message}")
        print(f"pipe called with user_message: {user_message}")

        # Handle title request
        if body.get("title", False):
            return "Compute Pipeline"

        # Verify the model is supported
        if model_id != "compute-bash":
            error_msg = f"Unsupported model ID: {model_id}. Use 'compute-bash'."
            logger.error(error_msg)
            return error_msg

        try:
            # Execute command using our enhanced loop
            output_parts = []

            # Define callbacks to collect output
            async def output_callback(content: Dict):
                if content["type"] == "text":
                    output_parts.append(content["text"])

            async def tool_callback(result, tool_id):
                if result.system:
                    output_parts.append(f"<SYSTEM>{result.system}</SYSTEM>")

                    # Run command through execute_command

            async def run_command():
                async for chunk in execute_command(
                    command=user_message,
                    model="claude-3-5-sonnet-20241022",
                    api_key=self.valves.ANTHROPIC_API_KEY,
                    messages=messages,  # Pass the message history
                    output_callback=output_callback,
                    tool_output_callback=tool_callback,
                ):
                    if chunk["content"]:
                        output_parts.append(chunk["content"])

            # Run the async function synchronously since pipe() is sync
            asyncio.run(run_command())

            # Return collected output
            return (
                "\n".join(output_parts)
                if output_parts
                else "Command executed successfully"
            )

        except Exception as e:
            error_msg = f"Error in pipe: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
