"""
Implementation of a bash shell tool that allows executing commands in a persistent session.

This module provides:
- _BashSession: Internal class managing the bash process and command execution
- BashTool: Main tool class implementing the Anthropic bash tool interface

The tool maintains a persistent bash session allowing for stateful operation (e.g. changing 
directories or setting environment variables that persist between commands). It handles
timeout, process management, and proper output/error capturing.

Example:
    bash_tool = BashTool()
    result = await bash_tool(command="ls -la")
    print(result.output)  # Directory listing
    
    # Reset the session if needed
    await bash_tool(restart=True)
"""

import asyncio
import os
from typing import ClassVar, Literal

from anthropic.types.beta import BetaToolBash20241022Param

from .base import BaseAnthropicTool, CLIResult, ToolError, ToolResult


class _BashSession:
    """Internal class that manages a persistent bash shell process.

    This class handles the lifecycle of a bash process including:
    - Starting and stopping the shell process
    - Executing commands and capturing their output
    - Managing timeouts and error conditions
    - Handling process cleanup

    The session maintains state between commands, so environment variables, current directory,
    and other shell state persists across command executions.

    Attributes:
        command (str): Path to the bash executable (defaults to /bin/bash)
        _output_delay (float): Delay between output checks in seconds
        _timeout (float): Maximum time to wait for command completion in seconds
        _sentinel (str): Marker string used to detect command completion

    The session uses a sentinel value to reliably detect when commands complete, as
    some commands may not immediately flush their output or may run in the background.
    """

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        """Start the bash shell process if it's not already running.

        Creates a new subprocess running bash with pipes for stdin/stdout/stderr.
        The process is started with setsid() to ensure proper process group management.
        No buffering is used to ensure immediate output availability.

        Multiple calls to start() on an already started session are ignored.
        """
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """Terminate the bash shell process.

        Sends SIGTERM to the process if it's still running. Does nothing if the
        process has already terminated.

        Raises:
            ToolError: If stop() is called before start()
        """
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """Execute a command in the bash shell and capture its output.

        The command is executed in the persistent shell session, allowing for
        stateful operations. A sentinel value is appended to detect completion.

        Args:
            command (str): The shell command to execute

        Returns:
            CLIResult: Contains the command's stdout in output and stderr in error

        Raises:
            ToolError: If the session isn't started or times out

        The method handles:
            - Command execution and output capture
            - Detection of command completion using a sentinel
            - Timeout management
            - Buffer clearing between commands
        """

        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = (
                        self._process.stdout._buffer.decode()
                    )  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error = (
            self._process.stderr._buffer.decode()
        )  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class BashTool(BaseAnthropicTool):
    """A tool that allows the agent to run bash commands through the Anthropic API.

    This tool implements the Anthropic bash tool interface (bash_20241022) which allows
    LLMs to execute bash commands in a persistent session. The tool maintains state
    between commands and provides proper error handling and resource cleanup.

    Features:
    - Persistent session management (environment variables, current directory persist)
    - Command execution with output/error capture
    - Session restart capability
    - Automatic resource cleanup
    - Timeout handling for long-running commands

    The tool parameters conform to the Anthropic API specification and cannot be modified:
    - name: Always "bash"
    - type: Always "bash_20241022"

    Example:
        tool = BashTool()
        # Run a command
        result = await tool(command="echo $PWD")
        print(result.output)  # Current directory

        # Restart the session if needed
        await tool(restart=True)

        # Run another command
        result = await tool(command="ls -l")
        print(result.output)  # Directory listing
    """

    _session: _BashSession | None
    name: ClassVar[Literal["bash"]] = "bash"
    api_type: ClassVar[Literal["bash_20241022"]] = "bash_20241022"

    def __init__(self):
        self._session = None
        super().__init__()

    async def __call__(
        self, command: str | None = None, restart: bool = False, **kwargs
    ):
        """Execute a command or restart the bash session.

        This method implements the main tool interface, allowing for command execution
        and session management. It handles:
        - Starting a new session if none exists
        - Restarting the session if requested
        - Executing commands and capturing their output

        Args:
            command: The bash command to execute, or None if restarting
            restart: If True, restart the session before executing the command
            **kwargs: Additional arguments (ignored)

        Returns:
            ToolResult: The result of the command execution, or a system message
                       for session restart

        Raises:
            ToolError: If no command is provided and not restarting

        Example:
            # Execute a command
            result = await tool(command="pwd")

            # Restart the session
            await tool(restart=True)

            # Execute after restart
            result = await tool(command="echo $HOME")
        """
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolResult(system="tool has been restarted.")

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("no command provided.")

    def to_params(self) -> BetaToolBash20241022Param:
        """Convert the tool configuration to Anthropic API parameters.

        Returns a dictionary containing the tool configuration that matches
        the Anthropic API specification for the bash_20241022 tool type.

        Returns:
            BetaToolBash20241022Param: Dictionary with tool type and name

        Note:
            The returned parameters are fixed as per Anthropic specification:
            {
                "type": "bash_20241022",
                "name": "bash"
            }
        """
        return {
            "type": self.api_type,
            "name": self.name,
        }
