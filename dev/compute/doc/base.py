"""
Core base classes for the tool system. This module defines the abstract interfaces and base
classes that all tools must implement.

The module provides:
- BaseAnthropicTool: Abstract base class for all Anthropic-compatible tools
- ToolResult: Data structure representing the result of a tool execution
- CLIResult: Specialized ToolResult for command-line outputs
- ToolFailure: Specialized ToolResult for failed executions 
- ToolError: Exception class for tool-related errors

Classes in this module establish the core contract that tools must follow to be compatible
with the Anthropic API and our pipeline system.
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import Any

from anthropic.types.beta import BetaToolUnionParam


class BaseAnthropicTool(metaclass=ABCMeta):
    """Abstract base class for Anthropic-defined tools.

    This class defines the interface that all tools must implement to be compatible with
    the Anthropic API and our pipeline system. Tools are executable components that can
    perform specific actions like running shell commands or manipulating files.

    Subclasses must implement:
    - __call__: The main execution method that runs the tool
    - to_params: Method to convert the tool configuration to Anthropic API parameters

    Example:
        class MyTool(BaseAnthropicTool):
            def __call__(self, **kwargs):
                # Tool implementation
                return ToolResult(output="Tool result")

            def to_params(self):
                return {"type": "my_tool", "name": "my_tool"}
    """

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """Executes the tool with the given arguments.

        This is the main execution method that subclasses must implement. It should
        contain the core logic of the tool's functionality.

        Args:
            **kwargs: Tool-specific arguments needed for execution

        Returns:
            ToolResult: The result of the tool's execution

        Raises:
            ToolError: If the tool encounters an error during execution
        """
        ...

    @abstractmethod
    def to_params(
        self,
    ) -> BetaToolUnionParam:
        """Converts the tool configuration to Anthropic API parameters.

        This method should return a dictionary containing the tool's configuration
        in a format compatible with the Anthropic API specifications.

        Returns:
            BetaToolUnionParam: A dictionary containing the tool's type and configuration
        """
        raise NotImplementedError


@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution.

    This immutable dataclass encapsulates the output of a tool execution, including
    any text output, errors, images, and system messages. It provides methods for
    combining results and checking if there is any content.

    Attributes:
        output (str | None): The main output text from the tool execution
        error (str | None): Any error message from the tool execution
        base64_image (str | None): Base64-encoded image data, if any
        system (str | None): System-level messages about the tool execution

    The class supports:
    - Boolean evaluation (True if any field has content)
    - Adding results together to combine outputs
    - Replacing field values to create modified results

    Example:
        result = ToolResult(output="Command output", error=None)
        if result:  # Will be True since output has content
            print(result.output)
    """

    output: str | None = None
    error: str | None = None
    base64_image: str | None = None
    system: str | None = None

    def __bool__(self):
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)


class CLIResult(ToolResult):
    """A specialized ToolResult for command-line interface outputs.

    This class is used specifically for tools that execute CLI commands and need to
    present their output in a CLI-appropriate format. The output will be rendered
    differently from regular tool results in the UI.

    Example:
        result = CLIResult(output="$ ls\nfile1.txt\nfile2.txt")
    """


class ToolFailure(ToolResult):
    """A specialized ToolResult that represents a failed execution.

    This class is used when a tool execution fails but can still return some information
    about the failure. Unlike ToolError which raises an exception, this allows for
    graceful handling of expected failure cases.

    Example:
        result = ToolFailure(error="File not found", system="Check file permissions")
    """


class ToolError(Exception):
    """Exception raised when a tool encounters an unrecoverable error.

    This exception should be raised when a tool encounters an error that prevents it
    from continuing execution or returning a meaningful result. Unlike ToolFailure,
    this represents an exceptional condition that needs immediate handling.

    Attributes:
        message (str): Detailed description of what caused the error

    Example:
        raise ToolError("Invalid input parameter: size must be positive")
    """

    def __init__(self, message):
        self.message = message
