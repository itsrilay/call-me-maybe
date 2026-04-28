"""Entry point for the call-me-maybe function calling tool.

This script parses command-line arguments, loads the necessary JSON
schemas and test prompts, initializes the LLM and the GenerationPipeline,
and processes each prompt to output the final constrained JSON calls.
"""
import argparse
import sys
from src.generation_pipeline import GenerationPipeline
from src.io_handler import IOHandler
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Main execution function for the CLI tool.

    Parses command-line arguments, initializes the IO handler to load
    and validate data, manages early-exit optimizations for empty inputs,
    orchestrates the LLM generation loop, and saves the final results.

    Raises:
        SystemExit: Exits with code 0 on successful empty-input handling
            or after successful generation. Exits with code 1 on errors.
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

    io = IOHandler()

    # Validate output path
    if not io.is_path_safe(args.output):
        print(
            f"Error: The output path '{args.output}' is forbidden. "
            "It targets a protected project file or directory.",
            file=sys.stderr
        )
        sys.exit(1)

    # Load and validate input
    fn_defs = io.load_functions(args.functions_definition)
    prompts = io.load_prompts(args.input)

    # Exit early if no prompts
    if not prompts:
        print("Input file is empty. Writing empty results and exiting.")
        io.save_results(args.output, [])
        sys.exit(0)

    # Prepare model and pipeline
    model = Small_LLM_Model()
    pipeline = GenerationPipeline(model, fn_defs)

    # Generation loop
    results = []
    for prompt_obj in prompts:
        prompt_text = prompt_obj.prompt
        print(f"\nProcessing: {prompt_text}")

        result = pipeline.run(prompt_text)

        if result is None:
            print(f"Skipping failed prompt: \"{prompt_text}\"")
            continue

        results.append(result.model_dump())
        print("\n" + "-"*30)

    io.save_results(args.output, results)


if __name__ == "__main__":
    main()
