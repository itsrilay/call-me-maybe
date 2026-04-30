"""Pydantic models for function definitions and structured outputs.

This module defines the schemas used to constrain the LLM decoding
process and validate the final JSON outputs.
"""
from typing import Any, Literal
from pydantic import BaseModel


class ParameterDetail(BaseModel):
    """Details of a single function parameter.

    Attributes:
        type: The data type of the parameter (e.g., "number", "string").
    """
    type: Literal["string", "number", "integer", "boolean", "null"]


class PromptInput(BaseModel):
    """Schema for a single natural language prompt from the input file.

    Attributes:
        prompt: The natural language request string.
    """
    prompt: str


class FunctionDefinition(BaseModel):
    """Schema definition for an available callable function.

    Attributes:
        name: The exact name of the function.
        description: A human-readable description of what the function does.
        parameters: A mapping of parameter names to their type details.
        returns: A dictionary specifying the return type of the function.
    """
    name: str
    description: str
    parameters: dict[str, ParameterDetail]
    returns: dict[str, str]


class FunctionCall(BaseModel):
    """Structured representation of the final generated function call.

    Attributes:
        prompt: The original natural language request.
        name: The name of the selected function.
        parameters: The extracted arguments required by the function
            (e.g., {"a": 2.0}).
    """
    prompt: str
    name: str
    parameters: dict[str, Any]
