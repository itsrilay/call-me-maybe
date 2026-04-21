from src.models import FunctionDefinition, FunctionCall
from src.JSONValidator import JSONValidator
from src.JSONFSM import JSONFSM
from src.common import StatesEnum
from llm_sdk import Small_LLM_Model
from pydantic import ValidationError
import json
import sys
import numpy as np


class GenerationPipeline:

    MAX_TOKENS = 256

    def __init__(
        self, model: Small_LLM_Model, fn_defs: list[FunctionDefinition]
    ) -> None:
        self.model = model
        self.fn_defs = fn_defs
        self.validator = JSONValidator(fn_defs)

        # Get vocabulary
        try:
            with open(f"{model.get_path_to_vocab_file()}") as file:
                self.vocabulary: list[str] = json.load(file)
        except FileNotFoundError:
            print("Couldn't parse LLM vocabulary", file=sys.stderr)
            sys.exit(1)

        self.decoded_vocabulary = [
            self.model.decode([i]) for i in range(len(self.vocabulary))
        ]

        self.token_to_id = {
            token: id for id, token in enumerate(self.vocabulary)
        }

        self.system_prompt = (
            "You are an expert AI assistant. Your task is to call"
            " a function to help the user.\n"
            "\nBased on the user's query, output ONLY a JSON "
            "object calling the correct function.\n"
            "You have access to the following functions:\n\n"
        )

        for fn in self.fn_defs:
            # Tell the model the function definitions
            self.system_prompt += f"{fn.name}:\n{fn.description}\n"
            parameters = "Parameters:\n"
            for param in fn.parameters:
                param_type = fn.parameters[param].type
                parameters += f"{param}: {param_type}\n"
            self.system_prompt += f"{parameters}\n"

        print(self.system_prompt)

    def run(self, user_question: str) -> FunctionCall | None:
        full_prompt = (
            f"{self.system_prompt}\n\n" +
            f"User prompt: {user_question}\n\n" +
            "Assistant: "
        )

        input_ids_list: list[int] = self.model.encode(full_prompt)[0].tolist()
        prompt_length = len(input_ids_list)
        tokens_generated = 0

        fsm = JSONFSM(self.fn_defs)

        while (
            fsm.state != StatesEnum.END and tokens_generated < self.MAX_TOKENS
        ):
            allowed_tokens = fsm.get_allowed_tokens(
                self.decoded_vocabulary, self.validator
            )

            logits = self.model.get_logits_from_input_ids(input_ids_list)

            logit_mask = np.full(len(logits), -float("inf"))

            logit_mask[allowed_tokens] = 0.0

            logits_arr = np.array(logits)

            final_scores = logits_arr + logit_mask

            token_id = int(np.argmax(final_scores))

            input_ids_list.append(token_id)

            token_string = self.model.decode([token_id])
            print(token_string, end="", flush=True)

            tokens_generated += 1

            fsm.update_state(token_string, self.validator)

        # Extract only the generated tokens
        generated_ids = input_ids_list[prompt_length:]

        # Turn them into a single string
        final_json_string = self.model.decode(generated_ids)

        try:
            result = json.loads(final_json_string)
            return FunctionCall(prompt=user_question, **result)
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"\nError processing prompt: {e}")
            return None
