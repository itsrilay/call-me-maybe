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
from src.models import FunctionDefinition
from src.GenerationPipeline import GenerationPipeline
from llm_sdk import Small_LLM_Model


MAX_TOKENS = 256


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
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error while loading {filepath}.")
        sys.exit(1)


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

    fn_defs: list[FunctionDefinition] = [
        FunctionDefinition(**fn) for fn
        in load_json_file(args.functions_definition)
    ]

    prompts: list[dict[str, str]] = [
        prompt for prompt
        in load_json_file(args.input)
    ]

    model = Small_LLM_Model()

    all_results = []
    pipeline = GenerationPipeline(model, fn_defs)

    for prompt in prompts:
        prompt_text = prompt["prompt"]
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
    os.makedirs(output_dir, exist_ok=True)

    # Save everything at the end
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
