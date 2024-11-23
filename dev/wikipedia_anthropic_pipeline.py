"""
title: Wikipedia Anthropic Combined Pipeline
author: Assistant
date: 2024-11-20
version: 1.0
license: MIT
description: A pipeline that fetches Wikipedia content and summarizes it using Anthropic's Claude
requirements: requests, sseclient-py
environment_variables: ANTHROPIC_API_KEY
"""

import os
import requests
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

from pipelines.wikipedia_pipeline import Pipeline as WikipediaPipeline
from pipelines.anthropic_manifold_pipeline import Pipeline as AnthropicPipeline


class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""

    def __init__(self):
        self.name = "Wikipedia Anthropic Combined Pipeline"
        
        # Initialize valves first
        self.valves = self.Valves(
            **{"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")}
        )
        
        # Initialize both sub-pipelines
        self.wiki_pipeline = WikipediaPipeline()
        self.anthropic_pipeline = AnthropicPipeline()
        
        # Explicitly set the Anthropic API key in the child pipeline
        self.anthropic_pipeline.valves.ANTHROPIC_API_KEY = self.valves.ANTHROPIC_API_KEY
        self.anthropic_pipeline.update_headers()  # Update headers with new key

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        await self.wiki_pipeline.on_startup()
        await self.anthropic_pipeline.on_startup()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        await self.wiki_pipeline.on_shutdown()
        await self.anthropic_pipeline.on_shutdown()

    async def on_valves_updated(self):
        print(f"on_valves_updated:{__name__}")
        # Update the Anthropic pipeline's API key when valves are updated
        self.anthropic_pipeline.valves.ANTHROPIC_API_KEY = self.valves.ANTHROPIC_API_KEY
        await self.anthropic_pipeline.on_valves_updated()  # This will update the headers

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        if body.get("title", False):
            return "Wikipedia Anthropic Pipeline"

        # 1. Get Wikipedia content
        wiki_content = self.wiki_pipeline.pipe(user_message, model_id, messages, body)
        
        if wiki_content == "No information found":
            return "Sorry, no Wikipedia information found for this query."

        # 2. Prepare prompt for Anthropic
        summary_prompt = f"Please provide a clear and concise summary of this Wikipedia article, highlighting the most important points: \n\n{wiki_content}"
        
        # 3. Create message structure for Anthropic
        anthropic_messages = [{"role": "user", "content": summary_prompt}]
        
        # 4. Get summary from Anthropic
        # Using claude-3-haiku by default for faster responses
        summary = self.anthropic_pipeline.pipe(
            summary_prompt,
            "claude-3-haiku-20240307",
            anthropic_messages,
            {
                "temperature": 0.7,
                "max_tokens": 1000,
            }
        )

        return summary
