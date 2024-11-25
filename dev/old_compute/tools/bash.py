"""
Bash tool implementation for compute pipeline.
"""

import asyncio
import os
import logging
from typing import ClassVar, Literal
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Base class for tool execution results"""

    output: str | None = None
    error: str | None = None
    system: str | None = None


@dataclass
class CLIResult(ToolResult):
    """Result from a CLI tool execution"""

    pass


class ToolError(Exception):
    """Base exception for tool errors"""

    pass


class _BashSession:
    """A session of a bash shell."""

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
        logger.debug("BashSession.start called")
        if self._started:
            logger.debug("Session already started")
            return

        logger.debug(f"Starting subprocess with command: {self.command}")
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
        logger.debug("Bash session started successfully")

    def stop(self):
        """Terminate the bash shell."""
        logger.debug("BashSession.stop called")
        if not self._started:
            logger.error("Attempting to stop session that hasn't started")
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            logger.debug("Process already terminated")
            return
        logger.debug("Terminating bash process")
        self._process.terminate()
        logger.debug("Bash process terminated")

    async def run(self, command: str):
        """Execute a command in the bash shell."""
        logger.debug(f"BashSession.run called with command: {command}")

        if not self._started:
            logger.error("Session has not started")
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            logger.error(f"Bash has exited with returncode {self._process.returncode}")
            return ToolResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            logger.error(f"Previous command timed out after {self._timeout} seconds")
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        logger.debug("Writing command to stdin")
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            logger.debug("Reading command output")
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    output = self._process.stdout._buffer.decode()
                    logger.debug(f"Current output buffer: {output}")
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            logger.error(f"Command timed out after {self._timeout} seconds")
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error = self._process.stderr._buffer.decode()
        if error.endswith("\n"):
            error = error[:-1]

        logger.debug(f"Command output: {output}")
        logger.debug(f"Command error: {error}")

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()
        self._process.stderr._buffer.clear()

        result = CLIResult(output=output, error=error)
        logger.debug(f"Returning result: {result}")
        return result


class BashTool:
    """Tool for running bash commands."""

    _session: _BashSession | None
    name: ClassVar[Literal["bash"]] = "bash"

    def to_params(self) -> dict:
        """Convert tool to Claude API format."""
        return {
            "name": "bash",
            "description": "Execute bash commands in the virtual machine",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    }
                },
                "required": ["command"],
            },
        }

    def __init__(self):
        self._session = None

    async def __call__(
        self, command: str | None = None, restart: bool = False, **kwargs
    ):
        logger.debug(
            f"BashTool called with command: {command}, restart: {restart}, kwargs: {kwargs}"
        )

        if restart:
            logger.debug("Restarting bash session")
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()
            return ToolResult(system="tool has been restarted.")

        if self._session is None:
            logger.debug("Initializing new bash session")
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            logger.debug(f"Running command: {command}")
            result = await self._session.run(command)
            logger.debug(f"Command result: {result}")
            return result

        logger.error("No command provided")
        raise ToolError("no command provided.")

    def sync_execute(self, command: str) -> ToolResult:
        """Execute a command synchronously using subprocess.run"""
        logger.debug(f"sync_execute called with command: {command}")

        try:
            import subprocess

            # Run command and capture output
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,  # Match async timeout
            )

            # Create result object
            result = CLIResult(
                output=process.stdout if process.stdout else None,
                error=process.stderr if process.stderr else None,
            )

            logger.debug(f"sync_execute result: {result}")
            return result

        except subprocess.TimeoutExpired:
            return CLIResult(
                error="Command timed out after 120 seconds", system="timeout"
            )
        except Exception as e:
            return CLIResult(error=str(e), system="error")
