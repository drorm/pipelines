"""
Core agentic loop implementation for Anthropic's computer use tool framework.

This module implements the main interaction loop between language models and computer
control tools through the Anthropic API. It provides the infrastructure for:

1. Language Model Integration
   - Support for multiple API providers (Anthropic, Bedrock, Vertex)
   - Managed conversation context and history
   - Prompt caching optimization
   - Error handling and recovery

2. Tool Management
   - Bash command execution
   - Computer GUI interaction
   - File system operations
   - Tool result processing and feedback

3. Performance Optimization
   - Image context pruning
   - Strategic prompt caching
   - Efficient message handling

The module is designed to be provider-agnostic while maintaining compatibility with
Anthropic's tool specifications. It handles the complexities of maintaining state,
managing resources, and providing a reliable interface between the LLM and computer
control tools.

Example Usage:
    ```python
    from computer_use_demo.loop import sampling_loop, APIProvider
    
    messages = [{
        "role": "user",
        "content": "What files are in the current directory?"
    }]
    
    result = await sampling_loop(
        model="claude-3-sonnet",
        provider=APIProvider.ANTHROPIC,
        messages=messages,
        output_callback=handle_output,
        tool_output_callback=handle_tool_output,
        api_response_callback=handle_api_response,
        api_key="your-key"
    )
    ```

The module provides a complete framework for building agentic applications that can
control and interact with computer systems through natural language instructions.
"""

import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

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

from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

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


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
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
    """Main agentic loop managing interactions between the LLM and tools.

    This function implements the core interaction loop between the language model
    and the available tools. It:
    1. Sends the current conversation state to the LLM
    2. Receives and processes the LLM's response
    3. Executes any tool actions requested by the LLM
    4. Feeds tool results back to the LLM
    5. Continues until the LLM completes its task or encounters an error

    Args:
        model: Name of the LLM model to use (e.g., "claude-3-sonnet")
        provider: The API provider to use (Anthropic, Bedrock, or Vertex)
        system_prompt_suffix: Additional context to append to the system prompt
        messages: The conversation history in Anthropic message format
        output_callback: Function to call with each LLM response block
        tool_output_callback: Function to call with tool execution results
        api_response_callback: Function to call with API request/response details
        api_key: API key for authentication (Anthropic only)
        only_n_most_recent_images: Optional limit on number of images to include
        max_tokens: Maximum tokens in LLM response (default: 4096)

    Returns:
        list[BetaMessageParam]: Updated conversation history including tool interactions

    The function handles:
    - Setting up tool collection (bash, computer, editor tools)
    - Managing conversation context including image pruning
    - API error handling and recovery
    - Tool execution and result integration
    - Prompt caching for improved performance

    Example:
        messages = [{"role": "user", "content": "List files in /tmp"}]
        result = await sampling_loop(
            model="claude-3",
            provider=APIProvider.ANTHROPIC,
            system_prompt_suffix="",
            messages=messages,
            output_callback=print_output,
            tool_output_callback=print_tool_output,
            api_response_callback=log_api_response,
            api_key="my-key"
        )
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )
    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    while True:
        enable_prompt_caching = False
        betas = [COMPUTER_USE_BETA_FLAG]
        image_truncation_threshold = 10
        if provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=api_key)
            enable_prompt_caching = True
        elif provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            client = AnthropicBedrock()

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(messages)
            # Is it ever worth it to bust the cache with prompt caching?
            image_truncation_threshold = 50
            system["cache_control"] = {"type": "ephemeral"}

        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                messages,
                only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )

        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
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


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """Filter the conversation history to keep only recent images.

    This function optimizes the conversation context by removing older screenshots,
    which typically become less relevant as the conversation progresses. It operates
    with the following assumptions:

    1. Screenshots (images) from earlier in the conversation are less important
    2. Removing images in chunks helps maintain prompt cache efficiency
    3. A minimum threshold of images should be removed to make the operation worthwhile

    Args:
        messages: The full conversation history to filter
        images_to_keep: Number of most recent images to retain
        min_removal_threshold: Minimum batch size for image removal

    The function modifies the messages list in-place, removing images from tool_result
    blocks while preserving all other content. Images are removed in batches of at
    least min_removal_threshold to optimize prompt cache usage.

    Example:
        # Keep only the 10 most recent images, removing in batches of 5
        _maybe_filter_to_n_most_recent_images(messages, 10, 5)
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _response_to_params(
    response: BetaMessage,
) -> list[BetaTextBlockParam | BetaToolUseBlockParam]:
    """Convert an Anthropic API response to message parameters.

    This function processes a response from the Anthropic API and converts it into
    a list of content blocks that can be used in the conversation history. It handles
    both text blocks and tool use blocks from the response.

    Args:
        response: The API response message to process

    Returns:
        list: A list of content blocks, each either text or tool use parameters

    The function:
    1. Extracts content blocks from the response
    2. Converts text blocks to simple text parameters
    3. Preserves tool use blocks as-is
    4. Returns all blocks in chronological order

    Example:
        response = client.messages.create(...)
        params = _response_to_params(response)
        messages.append({"role": "assistant", "content": params})
    """
    res: list[BetaTextBlockParam | BetaToolUseBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            res.append({"type": "text", "text": block.text})
        else:
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """Optimize API performance by configuring prompt caching breakpoints.

    This function manages the Anthropic API's prompt caching feature by strategically
    setting cache breakpoints in the conversation history. It implements a caching
    strategy that:

    1. Marks the 3 most recent conversation turns as ephemeral (not cached)
    2. Allows all earlier turns to be cached
    3. Reserves one cache slot for system prompt and tools, shared across sessions

    Args:
        messages: The conversation history to modify

    The function modifies the messages in-place, adding or removing cache control
    parameters as needed. Only user messages with tool results can be marked as
    cache breakpoints.

    Caching Strategy:
    - Recent turns are marked ephemeral to allow for dynamic conversation
    - Older turns are cached to improve API performance
    - System/tools share a cache to maintain consistency across sessions

    Example:
        messages = [...]  # Conversation history
        _inject_prompt_caching(messages)
        # Messages now have appropriate cache control settings
    """

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
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
