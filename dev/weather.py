"""
title: Weather Pipeline with Anthropic Claude
author: Assistant
date: 2024-11-20
version: 1.0
license: MIT
description: A pipeline that uses Claude to extract location from natural language and fetches weather data
requirements: requests
environment_variables: ANTHROPIC_API_KEY, WEATHER_API_KEY
"""

import os
import json
import requests
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

from pipelines.anthropic_manifold_pipeline import Pipeline as AnthropicPipeline

class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""
        WEATHER_API_KEY: str = ""

    def __init__(self):
        self.name = "Weather Pipeline"
        
        # Initialize valves
        self.valves = self.Valves(
            **{
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "your-api-key-here"),
                "WEATHER_API_KEY": os.getenv("WEATHER_API_KEY", "your-weather-api-key-here")
            }
        )
        
        # Initialize Anthropic pipeline
        self.anthropic_pipeline = AnthropicPipeline()
        self.anthropic_pipeline.valves.ANTHROPIC_API_KEY = self.valves.ANTHROPIC_API_KEY
        self.anthropic_pipeline.update_headers()

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

    def get_weather(self, location: str, unit: str = "celsius") -> str:
        """Get weather data from OpenWeatherMap API"""
        try:
            # Parse location to get city and country
            parts = location.split(',')
            city = parts[0].strip()
            country = parts[1].strip() if len(parts) > 1 else ''
            
            # Build API URL
            api_key = "44b5c9a1f54835720cf823b8f6928a82"  # Your provided API key
            units = "metric" if unit == "celsius" else "imperial"
            base_url = "https://api.openweathermap.org/data/2.5/weather"
            
            # Make API request
            params = {
                'q': f"{city},{country}",
                'APPID': api_key,
                'units': units
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise exception for bad status codes
            
            weather_data = response.json()
            temperature = weather_data['main']['temp']
            description = weather_data['weather'][0]['description']
            
            return f"{temperature:.1f} degrees {unit} with {description}"
            
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return "Sorry, I couldn't fetch the weather data at the moment."
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing weather data: {e}")
            return "Sorry, there was an error processing the weather data."

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        if body.get("title", False):
            return "Weather Pipeline"

        # Define the weather tool for Claude
        weather_tool = {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature to use"
                    }
                },
                "required": ["location"]
            }
        }

        # 1. Ask Claude to extract location from user message
        location_prompt = f"Extract the location from this question and format it as city, state/country. Question: {user_message}"
        location_messages = [{"role": "user", "content": location_prompt}]
        location_response = self.anthropic_pipeline.pipe(
            location_prompt,
            "claude-3-haiku-20240307",
            location_messages,
            {"temperature": 0.7, "max_tokens": 100}
        )

        # 2. Get weather for the location
        weather_data = self.get_weather(location_response, "celsius")

        # 3. Ask Claude to format the response nicely
        final_prompt = f"The weather in {location_response} is {weather_data}. Please format this into a natural, friendly response."
        final_messages = [{"role": "user", "content": final_prompt}]
        final_response = self.anthropic_pipeline.pipe(
            final_prompt,
            "claude-3-haiku-20240307",
            final_messages,
            {"temperature": 0.7, "max_tokens": 150}
        )

        return final_response