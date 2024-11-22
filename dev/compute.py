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
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

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

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        if body.get("title", False):
            return "Compute Pipeline"

        # Verify the model is supported
        if model_id != "compute-bash":
            raise ValueError(f"Unsupported model ID: {model_id}. Use 'compute-bash'.")

        # Extract and execute bash command from the user message
        try:
            result = self.bash_tool.run(user_message)
            return str(result) if result else "Command executed successfully"
        except Exception as e:
            return f"Error executing command: {str(e)}"