# Overview

## Goal

Migrate the "claude computer use demo" from Streamlit to Open WebUI Pipelines. 
- Use compute.py to replace streamlit.py as the main app.
- try to keep as much of the existing code as possible, so that we can benefit from ongoing improvements to the claude codebase. Ideally, we should just replace streamlit.py with compute.py and make minimal changes to the rest of the code.

In this directory we have the original code from "claude computer use demo" with slight modifications. 

.
├── __init__.py
├── streamlit.py -- the streamlit main app that serves as the UI and drives the app by calling loop.py
├── loop.py -- the main loop that runs the app and makes multiple calls to the vm and to claude
├── compute.py -- The new pipeline app that needs to run on open webui pipelines to replace the streamlit app
├── loggers.py -- the logger for the app. 
└── tools
    ├── __init__.py
    ├── base.py
    ├── bash.py
    ├── collection.py
    ├── computer.py
    ├── edit.py
    └── run.py


The main files we are interested in are

# streamlit.py 

This file implements the Streamlit-based web interface for the computer use demo. Key components include:

Main UI Setup and Configuration:
	Implements a chat interface using Streamlit's components
	Handles API provider selection (Anthropic, Bedrock, Vertex)
	Manages configuration settings (API keys, model selection, system prompts)
	Provides image display controls and system prompt customization

State Management:
	Uses Streamlit's session state to maintain conversation history
	Manages tool responses and API exchanges
	Handles authentication state for different providers

Message Handling:
	Processes and renders different message types (user, assistant, tool outputs)
	Handles API responses and error conditions
	Manages image rendering and screenshot display

Security Features:
	Includes security warnings about sensitive data
	Implements API key storage and validation
	Provides secure configuration file handling

- We can ignore a lot of the UI related parts.
- The main part of the code is the loop sampling_loop() function that calls the loop.py file.

# loop.py
loop.py Summary: This file implements the core agent loop for interacting with language models and tools. Key components include:

Core Loop Implementation:
	Manages the interaction between LLMs and computer control tools
	Supports multiple API providers (Anthropic, Bedrock, Vertex)
	Handles conversation context and tool execution

Tool Integration:
	Sets up and manages tool collection (Computer, Bash, Edit tools)
	Processes tool results and feeds them back to the LLM
	Converts tool outputs into API-compatible formats

Performance Optimization:
	Implements prompt caching strategies
	Manages image context pruning
	Optimizes conversation history handling

Error Handling:
	Manages API errors and recovery
	Handles tool execution failures
	Provides detailed error reporting

Key Features:

Configurable system prompts
Efficient handling of image-heavy conversations
Robust error handling and recovery
Optimized prompt caching for better performance

- This is the main loop that runs the app.
- It acts as an agent that
- calls claude with the initial request
- performs actions in the VM based on the response from claude
- reports back to claude with the results of the actions
- repeats the process until the action is complete, it fails or it is cancelled by the user.

# tools/base.py

Defines the core framework for the tool system through these key components:

BaseAnthropicTool (Abstract Base Class)
	The foundation class all tools must inherit from
	Requires implementation of two methods:
		__call__: Executes the tool
		to_params: Converts tool config to Anthropic API format

ToolResult (Dataclass)
	Standard container for tool execution results
	Contains fields for: output, error, base64_image, and system messages
	Immutable, with ability to combine results and check for content

Three specialized result classes:
	CLIResult: For command-line outputs
	ToolFailure: For expected failure cases
	ToolError: Exception class for critical failures

The file essentially establishes the contract that all tools must follow to be compatible with both the Anthropic API and the pipeline system.

# tools/bash.py

Core Implementation:
	Provides a persistent bash shell tool implementation
	Maintains session state between commands
	Implements the Anthropic bash tool interface

Key Components: a) _BashSession (Internal Class):
	Manages the bash process lifecycle
	Features:
		Async command execution
		Output/error capture
		Timeout handling
		Process cleanup
	Uses sentinel values to detect command completion
	Default timeout: 120 seconds

b) BashTool (Main Tool Class):
	Implements BaseAnthropicTool
	Fixed parameters:
		name: "bash"
		type: "bash_20241022"
	Capabilities:
		Command execution
		Session management
		Restart functionality
		Resource cleanup
