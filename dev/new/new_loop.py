"""
Agentic sampling loop that calls the Anthropic API and executes bash commands using function calling.

This module implements the core interaction loop between the LLM (Claude) and the bash tool.
It handles:
1. Message management and formatting 
2. Tool execution and result processing
3. Error handling and response callbacks
"""

import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, Union, AsyncGenerator, cast

import httpx
from anthropic import (
    Anthropic,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tools import BashTool, ToolCollection, ToolResult

COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
}


# This system prompt is optimized for the bash-only environment
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing an Ubuntu virtual machine using {platform.machine()} architecture with bash command execution capabilities
* You can execute any valid bash command but do not install packages
* When using commands that are expected to output very large quantities of text, redirect into a tmp file
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}
</SYSTEM_CAPABILITY>

<IMPORTANT> 
* Only execute valid bash commands
* Dangerous commands will return an error instead of executing
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
    max_tokens: int = 1024,
    max_operations: int = 5,
) -> AsyncGenerator[Dict[str, str], None]:
    """
    Execute commands through Claude to accomplish a given task, with up to max_operations operations.

    Args:
        command: The task/goal to accomplish
        model: The Claude model to use
        api_key: Anthropic API key
        messages: Optional list of previous message history
        output_callback: Optional callback for Claude's responses
        tool_output_callback: Optional callback for tool outputs
        max_tokens: Maximum number of tokens for model response
        max_operations: Maximum number of operations allowed

    Returns:
        An async generator yielding chunks of output as {"content": str} dictionaries
    """
    # Initialize bash tool
    bash_tool = BashTool()

    # Keep ToolCollection for compatibility but only use bash
    tool_collection = ToolCollection(bash_tool)

    # Create Anthropic client
    client = Anthropic(api_key=api_key)

    # Initialize operation counter
    operation_count = 0

    try:
        # Get Claude's interpretation/validation of the command
        if messages is None:
            messages = []

        # Add the initial task to messages
        messages.append({"role": "user", "content": command})

        while operation_count < max_operations:
            # Call the API with raw response for compatibility
            try:
                raw_response = client.beta.messages.with_raw_response.create(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=model,
                    system=SYSTEM_PROMPT,
                    tools=tool_collection.to_params(),
                )
            except (APIStatusError, APIResponseValidationError) as e:
                yield {"content": f"API Error: {str(e)}"}
                return
            except APIError as e:
                yield {"content": f"API Error: {str(e)}"}
                return

            response = raw_response.parse()
            response_params = _response_to_params(response)

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
                    assistant_content.append(content_block)

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


async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
):
    """
    Agentic sampling loop for the assistant/tool interaction.
    Preserved for backward compatibility and future enhancements.
    Currently only supports bash tool.
    """
    # Initialize bash tool with collection wrapper
    bash_tool = BashTool()
    tool_collection = ToolCollection(bash_tool)

    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    enable_prompt_caching = True
    betas = [COMPUTER_USE_BETA_FLAG]

    client = Anthropic(api_key=api_key)

    if enable_prompt_caching:
        betas.append(PROMPT_CACHING_BETA_FLAG)
        _inject_prompt_caching(messages)
        # Use caching parameters
        system["cache_control"] = {"type": "ephemeral"}

    # Call the API
    try:
        raw_response = client.beta.messages.with_raw_response.create(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            system=[system],
            tools=tool_collection.to_params(),
            betas=betas,
        )
    except (APIStatusError, APIResponseValidationError) as e:
        api_response_callback(e.request, e.response, e)
        return messages
    except APIError as e:
        api_response_callback(e.request, e.body, e)
        return messages

    api_response_callback(
        raw_response.http_response.request, raw_response.http_response, None
    )

    response = raw_response.parse()
    response_params = _response_to_params(response)
    messages.append(
        {
            "role": "assistant",
            "content": response_params,
        }
    )

    tool_result_content: list[BetaToolResultBlockParam] = []
    for content_block in response_params:
        output_callback(content_block)
        if content_block["type"] == "tool_use":
            result = await bash_tool(command=content_block["input"]["command"])
            tool_result_content.append(
                _make_api_tool_result(result, content_block["id"])
            )
            tool_output_callback(result, content_block["id"])

    if not tool_result_content:
        return messages

    messages.append({"content": tool_result_content, "role": "user"})


def _response_to_params(
    response: BetaMessage,
) -> list[BetaTextBlockParam | BetaToolUseBlockParam]:
    """Convert a Claude API response into a standardized parameter format."""
    res: list[BetaTextBlockParam | BetaToolUseBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            res.append({"type": "text", "text": block.text})
        else:
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert a tool execution result into Claude API-compatible format."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False

    if result.error:
        is_error = True
        output_text = result.error
    else:
        output_text = result.output if result.output else ""

    if output_text:
        tool_result_content = {
            "type": "text",
            "text": _maybe_prepend_system_tool_result(result, output_text),
        }

    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str) -> str:
    """Format tool result text with any system messages."""
    if not result_text:
        return ""

    if result.system:
        return f"<s>{result.system}</s>\n\n{format_tool_output(result)}"
    return format_tool_output(result)


def format_tool_output(result: ToolResult) -> str:
    """Format the tool output based on output/error content."""
    if result.output and result.error:
        return f"```\n{result.output}\n```\n{result.error}"

    if result.error:
        return result.error

    if result.output:
        return f"```\n{result.output}\n```"

    return ""


def _inject_prompt_caching(messages: list[BetaMessageParam]):
    """Inject cache control for prompt optimization."""
    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                break
