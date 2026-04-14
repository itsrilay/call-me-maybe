import argparse
import json
import os
from typing import Any
from src.models import FunctionDefinition, FunctionCall
from src.JSONFSM import JSONFSM
from src.JSONValidator import JSONValidator
from src.common import StatesEnum
from llm_sdk import Small_LLM_Model

MAX_TOKENS = 50


def load_json_file(filepath: str) -> list[Any]:
    with open(filepath) as file:
        return json.load(file)


def main() -> None:
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

    # Prompt to give the LLM context
    system_prompt = "You have access to the following functions:\n\n"

    for fn in fn_defs:
        arguments = ""
        for param in fn.parameters:
            param_type = fn.parameters[param].type
            arguments += f"{param}: {param_type}\n"
        system_prompt += f"{fn.name}:\n{fn.description}\n{arguments}"

    prompts: list[dict[str, str]] = [
        prompt for prompt
        in load_json_file(args.input)
    ]

    model = Small_LLM_Model()
    fsm = JSONFSM(fn_defs)
    validator = JSONValidator(fn_defs)

    vocabulary: list[str] = [
        vocab for vocab
        in load_json_file(model.get_path_to_vocab_file())
    ]

    token_to_id = {token: id for id, token in enumerate(vocabulary)}

    # Starting with the first prompt for testing
    prompt_text = prompts[0]["prompt"]

    # Append system prompt
    full_prompt = (
        f"{system_prompt}\n\nUser Question: {prompt_text}" +
        "\nAnswer in JSON format: "
    )

    # Get the tensor and convert the first row to a Python list
    input_ids_list = model.encode(full_prompt)[0].tolist()

    prompt_length = len(input_ids_list)

    tokens_generated = 0

    while fsm.state != StatesEnum.END and tokens_generated < MAX_TOKENS:
        allowed_tokens = fsm.get_allowed_tokens(vocabulary, validator)

        logit_mask: list[float] = [
            -float("inf") for _ in vocabulary
        ]

        for token in allowed_tokens:
            token_id = token_to_id[token]
            logit_mask[token_id] = 0.0

        logits = model.get_logits_from_input_ids(input_ids_list)

        final_scores = [
            logit + mask for logit, mask in zip(logits, logit_mask)
        ]

        pick = final_scores.index(max(final_scores))

        input_ids_list.append(pick)

        pick_string = model.decode([pick])
        print(pick_string, end="", flush=True)

        tokens_generated += 1

        fsm.update_state(pick_string, validator)

    all_results = []

    # Extract only the generated tokens
    generated_ids = input_ids_list[prompt_length:]

    # Turn them into a single string
    final_json_string = model.decode(generated_ids)

    result = json.loads(final_json_string)

    function_call = FunctionCall(prompt=prompt_text, **result)

    all_results.append(function_call.model_dump())

    # Get the directory path from the full file path
    output_dir = os.path.dirname(args.output)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save everything at the end
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
