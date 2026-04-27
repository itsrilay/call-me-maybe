"""Entry point for the call-me-maybe function calling tool.

This script parses command-line arguments, loads the necessary JSON
schemas and test prompts, initializes the LLM and the GenerationPipeline,
and processes each prompt to output the final constrained JSON calls.
"""
import argparse
import json
import sys
import os
from typing import Any
from src.models import FunctionDefinition, PromptInput
from src.GenerationPipeline import GenerationPipeline
from llm_sdk import Small_LLM_Model
from pydantic import ValidationError, TypeAdapter


def load_json_file(filepath: str) -> list[Any]:
    """Loads and parses a JSON file.

    Args:
        filepath (str): The path to the JSON file to be loaded.

    Returns:
        list[Any]: The parsed JSON content, typically a list of dictionaries.

    Raises:
        SystemExit: If the file is not found or contains invalid JSON.
    """
    try:
        with open(filepath) as file:
            data: list[Any] = json.load(file)
            if not isinstance(data, list):
                print(
                    f"Error: {filepath} must contain a JSON array.",
                    file=sys.stderr
                )
                sys.exit(1)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error while loading {filepath}.")
        sys.exit(1)


def format_pydantic_error(e: ValidationError) -> str:
    """Formats a Pydantic ValidationError into a readable string.

    Args:
        e (ValidationError): The caught Pydantic exception.

    Returns:
        str: A simplified error message focusing on the missing/invalid fields.
    """
    messages = []
    for error in e.errors():
        # Field path is a tuple, e.g., ('prompt',)
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        # Include the input that caused the issue for better debugging
        input_val = error.get("input")

        messages.append(
            f" - Field '{field}': {message} (Received: {input_val})"
        )

    return "\n".join(messages)


def main() -> None:
    """Main execution function for the CLI tool.

    Parses arguments, sets up the LLM pipeline, iterates over all provided
    user prompts, runs the constrained decoding generation, and writes the
    structured output to the specified destination file.
    """
    parser = argparse.ArgumentParser(
        prog="call-me-maybe",
        description="Function calling tool"
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json"
    )

    args = parser.parse_args()

    # Validate function definitions
    try:
        fn_adapter = TypeAdapter(list[FunctionDefinition])
        raw_fn_data = load_json_file(args.functions_definition)
        fn_defs = fn_adapter.validate_python(raw_fn_data)
    except ValidationError as e:
        print(
            "Error: Invalid function definition schema:\n",
            format_pydantic_error(e),
            file=sys.stderr
        )
        sys.exit(1)

    # Validate prompts
    try:
        prompt_adapter = TypeAdapter(list[PromptInput])
        raw_prompt_data = load_json_file(args.input)
        prompts = prompt_adapter.validate_python(raw_prompt_data)
    except ValidationError as e:
        print(
            "Error: Invalid prompt input schema:\n",
            format_pydantic_error(e),
            file=sys.stderr
        )
        sys.exit(1)

    model = Small_LLM_Model()

    all_results = []
    pipeline = GenerationPipeline(model, fn_defs)

    for prompt_obj in prompts:
        prompt_text = prompt_obj.prompt
        print(f"\nProcessing: {prompt_text}")

        result = pipeline.run(prompt_text)

        if result is None:
            print("Skipping failed prompt.")
            continue

        all_results.append(result.model_dump())
        print("\n" + "-"*30)

    # Get the directory path from the full file path
    output_dir = os.path.dirname(args.output)

    # Create the directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Save everything at the end
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=4)
    except OSError as e:
        print(
            f"Error: Could not write output to {args.output}: {e}",
            file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
