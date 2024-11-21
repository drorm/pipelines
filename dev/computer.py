"""
title: Computer/Bash Command Pipeline
author: Assistant
date: 2024-11-20
version: 1.0
license: MIT
description: A pipeline that allows LLMs to run bash commands and interact with the system
requirements: 
environment_variables: ANTHROPIC_API_KEY
"""

import os
import json
import subprocess
from typing import List, Union, Generator, Iterator, Any
from pydantic import BaseModel
from dataclasses import dataclass

from pipelines.anthropic_manifold_pipeline import Pipeline as AnthropicPipeline
from anthropic.types.beta import BetaToolUnionParam

@dataclass
class ToolResult:
    output: str | None = None
    error: str | None = None
    system: str | None = None

class BaseTool:
    @staticmethod
    def to_params() -> BetaToolUnionParam:
        raise NotImplementedError

class BashTool(BaseTool):
    def __call__(self, command: str | None = None, restart: bool = False) -> ToolResult:
        """Execute a bash command and return the result"""
        if restart:
            return ToolResult(system="Tool restarted.")

        if not command:
            return ToolResult(error="No command provided.")

        try:
            # Add basic command validation here
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return ToolResult(output=result.stdout)
            else:
                return ToolResult(error=f"Command failed with code {result.returncode}: {result.stderr}")

        except subprocess.TimeoutExpired:
            return ToolResult(error="Command timed out after 30 seconds")
        except Exception as e:
            return ToolResult(error=str(e))

    @staticmethod
    def to_params() -> BetaToolUnionParam:
        return {
            "type": "bash_20241022",
            "name": "bash"
        }

class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""

    def __init__(self):
        self.name = "Computer Pipeline"
        
        # Initialize valves
        self.valves = self.Valves(
            **{
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "your-api-key-here"),
            }
        )
        
        # Initialize Anthropic pipeline
        self.anthropic_pipeline = AnthropicPipeline()
        self.anthropic_pipeline.valves.ANTHROPIC_API_KEY = self.valves.ANTHROPIC_API_KEY
        self.anthropic_pipeline.update_headers()

        # Initialize tools
        self.bash_tool = BashTool()

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        await self.anthropic_pipeline.on_startup()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        await self.anthropic_pipeline.on_shutdown()

    async def on_valves_updated(self):
        print(f"on_valves_updated:{__name__}")
        self.anthropic_pipeline.valves.ANTHROPIC_API_KEY = self.valves.ANTHROPIC_API_KEY
        await self.anthropic_pipeline.on_valves_updated()

    def _make_tool_result(self, result: ToolResult, tool_use_id: str) -> dict:
        """Convert a ToolResult to the API expected format"""
        is_error = bool(result.error)
        content = []
        
        if result.system:
            content.append({"type": "text", "text": f"<system>{result.system}</system>"})
        
        if result.error:
            content.append({"type": "text", "text": result.error})
        elif result.output:
            content.append({"type": "text", "text": result.output})

        return {
            "type": "tool_result",
            "content": content,
            "tool_use_id": tool_use_id,
            "is_error": is_error,
        }

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        if body.get("title", False):
            return "Computer Pipeline"

        # Get tool response
        response = self.anthropic_pipeline.pipe(
            user_message,
            "claude-3-haiku-20240307",
            messages,
            {
                "temperature": 0.7,
                "max_tokens": 1024,
                "tools": [self.bash_tool.to_params()],
                "betas": ["computer-use-2024-10-22"]
            }
        )

        messages.append({"role": "assistant", "content": response})

        # Parse tool usage from response
        if isinstance(response, list):
            for block in response:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_name = block.get("name")
                    tool_input = block.get("input", {})
                    tool_id = block.get("id")

                    if tool_name == "bash":
                        result = self.bash_tool(**tool_input)
                        tool_result = self._make_tool_result(result, tool_id)
                        messages.append({"role": "user", "content": [tool_result]})

                        # Get final response after tool use
                        final_response = self.anthropic_pipeline.pipe(
                            user_message,
                            "claude-3-haiku-20240307", 
                            messages,
                            {
                                "temperature": 0.7,
                                "max_tokens": 1024,
                                "tools": [self.bash_tool.to_params()],
                                "betas": ["computer-use-2024-10-22"]
                            }
                        )
                        return final_response

        return response