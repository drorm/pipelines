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
from datetime import datetime
import platform
from typing import List, Union, Generator, Iterator, Any
from pydantic import BaseModel

from pipelines.anthropic_manifold_pipeline import Pipeline as AnthropicPipeline
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlockParam,
    BetaToolUseBlockParam,
    BetaToolResultBlockParam,
    BetaMessage,
)

from ..utils.tools import BashTool, ToolCollection, ToolResult

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

        # Initialize tool collection
        self.tool_collection = ToolCollection(BashTool())

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

    def _make_tool_result(self, result: ToolResult, tool_use_id: str) -> BetaToolResultBlockParam:
        """Convert a ToolResult to the API expected format"""
        tool_result_content: list[BetaContentBlockParam] = []
        is_error = bool(result.error)

        if result.system:
            tool_result_content.append({
                "type": "text",
                "text": f"<s>{result.system}</s>"
            })

        if result.error:
            tool_result_content.append({
                "type": "text",
                "text": result.error
            })
        elif result.output:
            tool_result_content.append({
                "type": "text",
                "text": result.output
            })

        return {
            "type": "tool_result",
            "content": tool_result_content,
            "tool_use_id": tool_use_id,
            "is_error": is_error,
        }

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"\n=== Starting pipe execution ===")
        print(f"User message: {user_message}")
        print(f"Model ID: {model_id}")
        print(f"Current messages count: {len(messages)}")
        print(f"Body: {json.dumps(body, indent=2)}")
        
        # Check if this is a tagging request
        if "### Task:" in user_message and "Generate 1-3 broad tags" in user_message:
            print("\nHandling tagging request...")
            return json.dumps({"tags": ["General", "Technology", "System"]})

        if body.get("title", False):
            return "Computer Pipeline"
            
        # Add system prompt if this is the first message
        if not messages:
            system_prompt = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
</IMPORTANT>"""

            messages.append({"role": "system", "content": system_prompt})
            print("Added system prompt to messages")

        # Get response from Anthropic
        try:
            response = self.anthropic_pipeline.pipe(
                user_message,
                model_id or "claude-3-5-sonnet-20241022",
                messages,
                {
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "tools": self.tool_collection.to_params(),
                    "betas": ["computer-use-2024-10-22"]
                }
            )
            print(f"\nInitial response: {json.dumps(response, indent=2)}")
        except Exception as e:
            print(f"\nError getting response from Anthropic: {e}")
            return f"Error: {str(e)}"

        # Handle tool usage if present
        if isinstance(response, list):
            for block in response:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_name = block.get("name")
                    tool_input = block.get("input", {})
                    tool_id = block.get("id")
                    
                    try:
                        result = self.tool_collection.run(tool_name, tool_input)
                        tool_result = self._make_tool_result(result, tool_id)
                        messages.append({"role": "user", "content": [tool_result]})
                        
                        # Get final response interpreting the results
                        print("\nGetting final response...")
                        final_response = self.anthropic_pipeline.pipe(
                            user_message,
                            model_id or "claude-3-5-sonnet-20241022",
                            messages,
                            {
                                "temperature": 0.7,
                                "max_tokens": 1024,
                                "tools": self.tool_collection.to_params(),
                                "betas": ["computer-use-2024-10-22"]
                            }
                        )
                        return final_response
                    except Exception as e:
                        print(f"\nError executing tool: {e}")
                        tool_result = self._make_tool_result(
                            ToolResult(error=str(e)),
                            tool_id
                        )
                        messages.append({"role": "user", "content": [tool_result]})

        return response
