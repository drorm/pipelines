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
    AnthropicBedrock,
    AnthropicVertex,
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
    BEDROCK = "bedrock"
    VERTEX = "vertex"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
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
    """
    tool_collection = ToolCollection(
        # ComputerTool(),
        BashTool(),
        # EditTool(),
    )
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
            result = await tool_collection.run(
                name=content_block["name"],
                tool_input=cast(dict[str, Any], content_block["input"]),
            )
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
