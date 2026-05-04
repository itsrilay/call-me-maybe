"""Entry point for the call-me-maybe function calling tool.

This script parses command-line arguments, loads the necessary JSON
schemas and test prompts, initializes the LLM and the GenerationPipeline,
and processes each prompt to output the final constrained JSON calls.
"""
import argparse
import sys
from src.generation_pipeline import GenerationPipeline
from src.io_handler import IOHandler
from src.json_validator import JSONValidator
from src.json_fsm import JSONFSM
from src.common import IOHandlerError, NoPromptsFound
from src.models import FunctionDefinition, PromptInput
from llm_sdk import Small_LLM_Model
import numpy as np
import numpy.typing as npt


def prepare_resources(args: argparse.Namespace, io: IOHandler) -> tuple[
    Small_LLM_Model,
    list[FunctionDefinition],
    list[PromptInput],
    list[str],
    JSONValidator,
    npt.NDArray[np.float32]
]:
    """Validates paths, loads all JSON resources, and prepares the logit mask.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        io (IOHandler): The handler used for file operations.

    Returns:
        tuple: A collection of all initialized objects required for generation.
    """
    # Validate output path
    if not io.is_path_safe(args.output):
        raise IOHandlerError(
            f"Error: The output path '{args.output}' is forbidden. "
            "It targets a protected project file or directory.",
        )

    # Load and validate input
    fn_defs = io.load_functions(args.functions_definition)
    prompts = io.load_prompts(args.input)

    # Prepare model and its vocabulary
    model = Small_LLM_Model()
    vocabulary = io.load_vocabulary(model.get_path_to_vocab_file())
    decoded_vocabulary = [model.decode([i]) for i in range(len(vocabulary))]

    # Prepare validator and logit mask
    validator = JSONValidator(fn_defs, decoded_vocabulary)

    dummy_logits = model.get_logits_from_input_ids([0])
    logit_size = len(dummy_logits)
    logit_mask = np.empty(logit_size, dtype=np.float32)

    return (model, fn_defs, prompts, decoded_vocabulary, validator, logit_mask)


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

    try:
        model, fn_defs, prompts, decoded_vocabulary, validator, logit_mask = (
            prepare_resources(args, io)
        )
    except IOHandlerError as e:
        print(f"Initialization Failed: {e}", file=sys.stderr)
        sys.exit(1)
    except NoPromptsFound as e:
        print(f"Warning: {e}", file=sys.stderr)
        io.save_results(args.output, [])
        sys.exit(0)

    pipeline = GenerationPipeline(
        model, fn_defs, decoded_vocabulary, validator, logit_mask
    )

    # Generation loop
    results = []
    for prompt_obj in prompts:
        prompt_text = prompt_obj.prompt
        print(f"\nProcessing: {prompt_text}")

        fsm = JSONFSM(pipeline.fn_defs)
        result = pipeline.run(prompt_text, fsm)

        if result is None:
            print(f"Skipping failed prompt: \"{prompt_text}\"")
            continue

        results.append(result.model_dump())
        print("\n" + "-"*30)

    io.save_results(args.output, results)


if __name__ == "__main__":
    main()
