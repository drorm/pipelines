#!/usr/bin/env python3
"""
CLI interface for testing compute.py functionality.
Usage: ./cli.py "your command here"
"""

import sys
import asyncio
import argparse
from typing import List, Dict
from pathlib import Path


from .compute import Pipeline


def create_mock_body() -> Dict:
    """Create a mock body for pipeline calls"""
    return {
        "temperature": 0.7,
        "stream": True,
    }


def create_mock_messages(command: str) -> List[Dict]:
    """Create initial message history with user command"""
    return [{"role": "user", "content": command}]


async def main(command: str):
    """Main CLI execution flow"""
    pipeline = Pipeline()

    # Create mock pipeline call parameters
    body = create_mock_body()
    messages = create_mock_messages(command)

    # Execute pipeline
    result = await pipeline.pipe(
        user_message=command, model_id="compute-bash", messages=messages, body=body
    )

    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test compute.py via CLI")
    parser.add_argument("command", help="Command to execute")
    args = parser.parse_args()

    asyncio.run(main(args.command))
