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
from typing import List, Union, Dict, Generator, Iterator
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a stream handler if one doesn't exist
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("Logger initialized for compute.py")

from dev.compute.loop import execute_command
from dev.compute.tools.bash import BashTool

class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""

    def __init__(self):
        self.name = "Compute Pipeline"
        self.type = "manifold"
        self.id = "compute"
        logger.debug(f"#### Initializing {self.name} pipeline")
        print(f"***** Initializing {self.name} pipeline *****")
        
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

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Execute commands through the compute pipeline."""
        print(f"pipe:{__name__}")  # Add debug print to match weather.py pattern
        
        try:
            # Handle title request
            if body.get("title", False):
                return "Compute Pipeline"

            # Verify the model is supported
            if model_id != "compute-bash":
                return f"Unsupported model ID: {model_id}. Use 'compute-bash'."

            # Execute command directly using BashTool
            result = self.bash_tool.sync_execute(command=user_message)
            
            # Format the output
            output_parts = []
            if result.output:
                output_parts.append(result.output)
            if result.error:
                output_parts.append(f"Error: {result.error}")
            if result.system:
                output_parts.append(f"System: {result.system}")
            
            return "\n".join(output_parts) if output_parts else "Command executed successfully"

        except Exception as e:
            error_msg = f"Error in pipe: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg