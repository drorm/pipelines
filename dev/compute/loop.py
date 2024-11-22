"""
Simple compute pipeline loop for executing bash commands.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any, Dict, Union, AsyncGenerator

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
    output_callback: Callable[[Dict[str, Any]], None] | None = None,
    tool_output_callback: Callable[[ToolResult, str], None] | None = None,
    stream_callback: Callable[[str], Any] | None = None,
) -> AsyncGenerator[Dict[str, str], None]:
    """
    Execute a single bash command and yield the results as chunks.
    
    Args:
        command: The bash command to execute
        model: The Claude model to use
        api_key: Anthropic API key
        output_callback: Optional callback for Claude's responses
        tool_output_callback: Optional callback for tool outputs
        stream_callback: Optional callback for streaming output
        
    Returns:
        An async generator yielding chunks of output as {"content": str} dictionaries
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
            yield {"content": "No response from Claude"}
            return
            
        parsed_command = response.content[0].text
        if output_callback:
            await output_callback({"type": "text", "text": parsed_command})
        
        # Execute the command
        result = await bash_tool(command=parsed_command)
        
        if tool_output_callback:
            await tool_output_callback(result, "bash-1")

        # Get the output text
        output = result.output if result.output else result.error if result.error else ""
        
        # Handle streaming if requested
        if stream_callback:
            async for chunk in stream_callback(output):
                yield chunk
        else:
            # For non-streaming case, wrap the output in a single chunk
            yield {"content": output}
        
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        yield {"content": error_msg}
