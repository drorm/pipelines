"""
title: Compute Pipeline with Bash Tool
author: Assistant
date: 2024-11-20
version: 1.0
license: MIT
description: A pipeline that enables execution of bash commands through an API endpoint
requirements: anthropic
environment_variables: ANTHROPIC_API_KEY
"""

import os
import json
import logging
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import compute functionality from the compute subdirectory
from dev.compute.tools.bash import BashTool

class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""
        
    def __init__(self):
        self.name = "Compute Pipeline"
        self.type = "manifold"
        self.id = "compute"
        
        # Initialize valves
        self.valves = self.Valves(
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", "")
        )
        
        # Initialize tools
        self.bash_tool = BashTool()

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        print(f"on_valves_updated:{__name__}")

    def pipelines(self) -> List[dict]:
        """Return list of supported models/pipelines"""
        return [{
            "id": "compute-bash",
            "name": "Compute Pipeline (Bash)",
            "description": "Execute bash commands via pipeline"
        }]

    async def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.debug(f"pipe called with user_message: {user_message}")
        logger.debug(f"model_id: {model_id}")
        logger.debug(f"messages: {messages}")
        logger.debug(f"body: {body}")

        # Handle title request
        if body.get("title", False):
            logger.debug("Returning title")
            return "Compute Pipeline"

        # Check for streaming mode
        stream_mode = body.get("stream", False)
        logger.debug(f"Stream mode: {stream_mode}")

        # Verify the model is supported
        if model_id != "compute-bash":
            logger.error(f"Unsupported model ID: {model_id}")
            raise ValueError(f"Unsupported model ID: {model_id}. Use 'compute-bash'.")

        try:
            # BashTool uses __call__ method which is async
            logger.debug(f"Attempting to execute command: {user_message}")
            result = await self.bash_tool(command=user_message)
            logger.debug(f"BashTool result: {result}")
            
            # Format the output nicely
            output_parts = []
            if result.output:
                logger.debug(f"Command output: {result.output}")
                output_parts.append(result.output)
            if result.error:
                logger.debug(f"Command error: {result.error}")
                output_parts.append(f"Error: {result.error}")
            if result.system:
                logger.debug(f"System message: {result.system}")
                output_parts.append(f"System: {result.system}")
            
            final_output = "\n".join(output_parts) if output_parts else "Command executed successfully"
            logger.debug(f"Final output: {final_output}")

            if stream_mode:
                # Return an async generator for streaming
                async def stream_output():
                    yield final_output
                return stream_output()
            else:
                # Return direct string response for non-streaming
                return final_output

        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if stream_mode:
                async def stream_error():
                    yield error_msg
                return stream_error()
            else:
                return error_msg
