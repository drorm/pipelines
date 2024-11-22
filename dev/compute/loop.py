"""
Simple compute pipeline loop for executing bash commands.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any, Dict

from anthropic import Anthropic
from .tools.bash import BashTool, ToolResult

SYSTEM_PROMPT = f"""You are a compute pipeline that executes bash commands.

<CAPABILITIES>
* You are utilizing an Ubuntu virtual machine for executing bash commands
* You can execute any valid bash command including installing packages
* When commands output large quantities of text, redirect to a tmp file
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}
</CAPABILITIES>

<IMPORTANT>
* Only execute the command provided, do not try to guess or expand the user's intent
* Do not try to use Firefox or X11 applications
* Focus on command execution and providing clear output
* If a command might be dangerous, return an error instead of executing it
</IMPORTANT>"""

async def execute_command(
    *,
    command: str,
    model: str,
    api_key: str,
    output_callback: Callable[[Dict[str, Any]], None],
    tool_output_callback: Callable[[ToolResult, str], None],
) -> str:
    """
    Execute a single bash command and return the result.
    
    Args:
        command: The bash command to execute
        model: The Claude model to use
        api_key: Anthropic API key
        output_callback: Callback for Claude's responses
        tool_output_callback: Callback for tool outputs
        
    Returns:
        The command output or error message
    """
    # Initialize bash tool
    bash_tool = BashTool()
    
    # Create Anthropic client
    client = Anthropic(api_key=api_key)
    
    try:
        # Get Claude's interpretation/validation of the command
        response = client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": command}],
            model=model,
            system=SYSTEM_PROMPT,
        )

        # Extract the command to execute
        if not response.content:
            return "No response from Claude"
            
        parsed_command = response.content[0].text
        output_callback({"type": "text", "text": parsed_command})

        # Execute the command
        result = await bash_tool(command=parsed_command)
        tool_output_callback(result, "bash-1")

        return result.output if result.output else result.error if result.error else ""
        
    except Exception as e:
        return f"Error executing command: {str(e)}"