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
from typing import List, Union, Dict, AsyncGenerator
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    ) -> str:
        """Get bash command without executing it."""
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
            # First, use Claude to convert natural language to bash command
            from dev.compute.claude import anthropic_completion
            
            prompt = f"""Based on this request: {user_message}
            Output ONLY the bash command that would execute this request.
            Do not execute it. Do not provide explanations.
            If multiple commands are needed, combine them with && or ;"""
            
            messages = [{"role": "user", "content": prompt}]
            bash_command = anthropic_completion(
                messages,
                self.valves.ANTHROPIC_API_KEY,
                "claude-3-5-sonnet-20241022",
                temperature=0,
                max_tokens=100
            ).strip()
            
            # Then execute the bash command
            result = self.bash_tool.sync_execute(command=bash_command)
            output_parts = []
            if result.output:
                output_parts.append(result.output)
            if result.error:
                output_parts.append(f"Error: {result.error}")
            if result.system:
                output_parts.append(f"System: {result.system}")
            
            # Return both the command and its output
            output_parts.insert(0, f"Executing: {bash_command}")
            return "\n".join(output_parts) if output_parts else "Command executed successfully"

        except Exception as e:
            error_msg = f"Error in pipe: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
