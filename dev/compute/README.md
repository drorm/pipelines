# Compute Pipeline

## Overview

The Compute Pipeline is an Open WebUI Pipeline implementation that provides command execution capabilities through an API endpoint. It migrates the functionality of the "claude computer use demo" from Streamlit into the Open WebUI Pipelines framework.

## Dependencies

Core dependencies:
- Python 3.11+
- Pydantic
- FastAPI
- Pygments (for automatic code language detection and syntax highlighting)

To install Pygments:
```bash
pip install pygments
```

## Project Structure

```
.
├── compute.py     -- Main pipeline implementation
├── loop.py        -- Core agent loop for LLM interaction
├── logger.py      -- Logging configuration
└── tools/
    ├── base.py    -- Base tool implementations
    ├── bash.py    -- Bash command execution tool
    └── ...
```

## Key Components

### compute.py

The main pipeline implementation that:
- Provides an OpenAI API-compatible endpoint
- Handles message streaming
- Manages tool execution
- Features automatic code language detection and syntax highlighting
- Supports both streaming and non-streaming responses

### loop.py

Core agent loop implementation that:
- Manages interactions between LLMs and computer control tools
- Supports multiple API providers (Anthropic, etc.)
- Handles conversation context and tool execution
- Implements prompt caching and optimization
- Provides robust error handling

### tools/bash.py

Bash tool implementation providing:
- Persistent bash shell sessions
- Async command execution
- Output/error capture with syntax highlighting
- Session state management
- Proper resource cleanup

## Features

### Core Features
- System prompt handling through message history
  - Extracts and applies system prompts from conversation context
  - Preserves system instructions across interactions

- Code Output Processing
  - Automatic programming language detection
  - Smart code fence wrapping (\`\`\`language)
  - Fallback to plain text for unknown content

- Streaming Support
  - Real-time response streaming
  - Async queue-based implementation
  - Support for both streaming and single-response modes

- Tool Integration
  - Structured tool output handling
  - Error code and message capture
  - Exit status reporting

- Model Support
  - Currently using Claude 3 Sonnet
  - Extensible to other API providers

## Usage

To use the pipeline:

1. Install dependencies:
```bash
pip install pygments
```

2. Configure the ANTHROPIC_API_KEY environment variable
3. The pipeline runs on port 9099 by default
4. Access through any OpenAI API-compatible client

## Security Considerations

- The pipeline has arbitrary code execution capabilities
- Only use in trusted environments
- Keep API keys secure
- Follow standard security practices when exposing endpoints

## Error Handling

The pipeline provides:
- Detailed error messages
- Exit code reporting
- Syntax-highlighted error output
- Proper cleanup on failures

## Logging

Comprehensive logging includes:
- Tool execution details
- API interactions
- Error conditions
- Performance metrics
