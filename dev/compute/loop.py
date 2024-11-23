"""
Agentic sampling loop that calls the Anthropic API and executes bash commands using function calling.

This module implements the core interaction loop between the LLM (Claude) and the bash tool.
It handles:
1. Message management and formatting 
2. Tool execution and result processing
3. Error handling and response callbacks
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any, Dict, Union, AsyncGenerator

from anthropic import Anthropic
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tools.bash import BashTool, ToolResult

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing an Ubuntu virtual machine with bash command execution capabilities
* You can execute any valid bash command including installing packages via apt
* When using commands that are expected to output very large quantities of text, redirect into a tmp file
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}
</SYSTEM_CAPABILITY>

<IMPORTANT> 
* Only execute valid bash commands
* Dangerous commands will return an error instead of executing
* Do not try to use X11/GUI applications
* For each task, you must either:
  1. Complete the requested goal and indicate success with "TASK COMPLETED: [brief explanation of what was accomplished]"
  2. Determine the goal cannot be achieved and indicate failure with "TASK FAILED: [explanation of why it cannot be done]"
* You have a maximum of 5 operations to complete each task
* Each command execution counts as one operation
* If you reach the operation limit without completing the task, respond with "OPERATION LIMIT REACHED: [current status and what remains to be done]"
</IMPORTANT>"""


async def execute_command(
    *,
    command: str,
    model: str,
    api_key: str,
    messages: list[dict] | None = None,
    output_callback: Callable[[Dict[str, Any]], None] | None = None,
    tool_output_callback: Callable[[ToolResult, str], None] | None = None,
) -> AsyncGenerator[Dict[str, str], None]:
    """
    Execute commands through Claude to accomplish a given task, with up to 5 operations.

    Args:
        command: The task/goal to accomplish
        model: The Claude model to use
        api_key: Anthropic API key
        messages: Optional list of previous message history
        output_callback: Optional callback for Claude's responses
        tool_output_callback: Optional callback for tool outputs

    Returns:
        An async generator yielding chunks of output as {"content": str} dictionaries
    """
    # Initialize bash tool
    bash_tool = BashTool()
    tool_collection = [bash_tool.to_params()]

    # Create Anthropic client
    client = Anthropic(api_key=api_key)

    # Initialize operation counter
    operation_count = 0
    max_operations = 5

    try:
        # Get Claude's interpretation/validation of the command
        if messages is None:
            messages = []

        # Add the initial task to messages
        messages.append({"role": "user", "content": command})

        while operation_count < max_operations:
            raw_response = client.beta.messages.create(
                max_tokens=1024,
                messages=messages,
                model=model,
                system=SYSTEM_PROMPT,
                tools=tool_collection,
            )

            response_params = _response_to_params(raw_response)

            # Track if this turn used a tool operation
            used_tool = False
            response_content = []  # Collect all content for this turn

            # First process and collect assistant's response
            tool_result_content = []  # Collect tool results for user message
            assistant_content = []  # Collect text/tool use for assistant message

            for content_block in response_params:
                if output_callback:
                    await output_callback(content_block)

                if content_block["type"] == "text":
                    text = content_block["text"]
                    assistant_content.append({"type": "text", "text": text})
                    yield {"content": text}

                    # Check for task completion markers
                    text_lower = text.lower()
                    if any(
                        marker in text_lower
                        for marker in [
                            "task completed:",
                            "task failed:",
                            "operation limit reached:",
                        ]
                    ):
                        if assistant_content:
                            messages.append(
                                {"role": "assistant", "content": assistant_content}
                            )
                        return

                elif content_block["type"] == "tool_use":
                    used_tool = True
                    assistant_content.append(
                        content_block
                    )  # Add tool use to assistant message

                    # Execute the command through the bash tool
                    result = await bash_tool(command=content_block["input"]["command"])

                    if tool_output_callback:
                        await tool_output_callback(result, content_block["id"])

                    # Process the tool result
                    tool_result = _make_api_tool_result(result, content_block["id"])
                    tool_result_content.append(tool_result)  # Collect for user message

                    # Get the output text with any system messages prepended
                    output = _maybe_prepend_system_tool_result(
                        result,
                        (
                            result.output
                            if result.output
                            else result.error if result.error else ""
                        ),
                    )

                    yield {"content": output}

            # Add assistant's response (text and tool uses)
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            # Add tool results as user message
            if tool_result_content:
                messages.append({"role": "user", "content": tool_result_content})

            # If we used a tool, increment the operation counter
            if used_tool:
                operation_count += 1
                if operation_count >= max_operations:
                    yield {
                        "content": "Operation limit reached. Waiting for LLM's final status..."
                    }

            # If no tool was used, the LLM is likely done
            if not used_tool:
                return

    except Exception as e:
        yield {"content": f"Error executing command: {str(e)}"}


def _response_to_params(
    response,
) -> list[BetaTextBlockParam | BetaToolUseBlockParam]:
    """
    Converts a Claude API response into a standardized parameter format.

    Args:
        response: Raw message response from Claude API call

    Returns:
        List of formatted content blocks, each either text or tool use parameters
    """
    res: list[BetaTextBlockParam | BetaToolUseBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            res.append({"type": "text", "text": block.text})
        else:
            res.append(block.model_dump())
    return res


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """
    Converts internal tool execution results into Claude API-compatible format.

    Args:
        result: The tool execution result to format
        tool_use_id: ID of the tool use block this result responds to

    Returns:
        Formatted tool result block
    """
    tool_result_content = []
    is_error = False

    if result.error:
        is_error = True
        output_text = result.error
    else:
        output_text = result.output if result.output else ""

    if output_text:
        tool_result_content.append(
            {
                "type": "text",
                "text": _maybe_prepend_system_tool_result(result, output_text),
            }
        )

    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str) -> str:
    """
    Prepends any system messages to tool output text.

    Args:
        result: The tool execution result
        result_text: The raw output/error text from the tool

    Returns:
        Text with any system messages prepended
    """
    if not result_text:
        return ""

    if result.system:
        return f"<s>{result.system}</s>\n\n{format_tool_output(result)}"
    return format_tool_output(result)


def format_tool_output(result: ToolResult) -> str:
    """
    Format the tool output appropriately based on whether it's output or error.

    Args:
        result: The tool execution result

    Returns:
        Formatted string with output/error
    """
    # If we have both output and error, format accordingly
    if result.output and result.error:
        return f"```\n{result.output}\n```\n{result.error}"

    # If we only have error, return it as plain text
    if result.error:
        return result.error

    # If we only have output, wrap it in code blocks
    if result.output:
        return f"```\n{result.output}\n```"

    return ""
