#!/usr/bin/env python3
"""
CLI interface for testing compute.py functionality.
Usage: ./cli.py "your command here"
"""

import sys
import asyncio
import argparse
import os
from typing import List, Dict
from pathlib import Path

# Add parent directory to Python path when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .compute import Pipeline  # Try relative import first
except ImportError:
    from compute import Pipeline  # Fallback to direct import


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
    result = pipeline.pipe(
        user_message=command, model_id="compute-bash", messages=messages, body=body
    )

    # Print each message in the stream
    for msg in result:
        print(msg, end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test compute.py via CLI")
    parser.add_argument("command", help="Command to execute")
    args = parser.parse_args()

    asyncio.run(main(args.command))
