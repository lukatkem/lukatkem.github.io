"""AgentCore — an LLM agent loop from first principles.

Tool registry with strict validation, a wire-format parser for fenced
tool calls, a scripted offline model, and the guarded loop that ties it
all together. Pure standard library, zero network calls.
"""
from .tools import (
    Registry,
    Tool,
    ToolError,
    calculator_tool,
    read_note_tool,
    text_stats_tool,
)
from .llm import BaseLLM, LLMResponse, MockLLM, ToolCallRequest
from .parser import ParsedToolCall, ParseError, parse_tool_calls
from .agent import Agent, AgentResult, Step, ToolResult

__version__ = "1.0.0"

__all__ = [
    "Agent",
    "AgentResult",
    "BaseLLM",
    "LLMResponse",
    "MockLLM",
    "ParsedToolCall",
    "ParseError",
    "Registry",
    "Step",
    "Tool",
    "ToolCallRequest",
    "ToolError",
    "ToolResult",
    "calculator_tool",
    "parse_tool_calls",
    "read_note_tool",
    "text_stats_tool",
]
